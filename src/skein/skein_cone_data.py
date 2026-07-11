"""`SkeinConeData` — the `ConeData` refinement for skein realizations.

The unified `SkeinKAlgebra` is a `ConeKAlgebra`; this module supplies
the cone-data shape that the skein side realises geometrically.  The
dictionary:

  * **mult-gens = RAYS** = simple, non-self-intersecting curves/arcs;
  * **Wilson lines / closed simple curves are SU(2)-character
    generators** (`char_gens` — the canonical basis on such a ray is
    the χ_k character tower, Clebsch–Gordan by construction);
    **open curves (arcs) are monomial generators** (parallel copies
    are canonical — the Lê stated-arc picture);
  * **`q_commute` = geometric disjointness** of the rays (two disjoint
    simple curves' canonicals q-commute);
  * **`cocycle` is defined ABSTRACTLY** — the σ̃ / Dirac pairing on the
    rays' charges, never a drawn-curve crossing count;
  * **`cross_product` is ENGINE-BACKED**: a certified skein engine
    (pinned polygon / cut-reglue / bracelet peel) computes `L_g·L_h`
    for crossing rays, and the result is re-expressed as literal cone
    words here.  The **standing guard** (coefficients asserted against
    the certified twin, honest-fail off scope — never fitted) lives in
    the engine, exactly as in the per-chart bridges.

Word/phase conventions (must match `cone_data.py`'s reducer exactly):
a literal in-order word `w` folds to `q^{-phase}·L_native` with
`phase = cone_label_phase(gens, powers)` (the `_reduction_step` leaf),
so a canonical engine term `coeff·L_lbl` becomes the word term
`(coeff·q^{+phase(lbl)}, word(lbl))` for monomial-cone labels, and the
`Cone.canonical_to_pbw` Chebyshev expansion for character-cone labels
(valid as literal words when the char cone's internal cocycles vanish
— asserted).
"""

from __future__ import annotations

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from abc import abstractmethod
from typing import Iterable, Sequence

from cone_data import Cone, ConeData, CrossProductTerm
from kalgebra import Element, Label


__all__ = ["SkeinConeData", "IntrinsicBridgeSkeinConeData",
           "fraction_solve"]


def fraction_solve(cols, rhs, dim=None):
    """The layer's ONE exact linear solver (this
    kernel was copy-pasted six times across the atlas/instance
    dictionaries with only postconditions differing — any change to
    the pivot or free-variable policy must land HERE, once).

    Solve `Σ_j x_j · cols[j] = rhs` over the rationals by Gaussian
    elimination with greedy column-order pivoting and **free columns
    zeroed** (the layer's canonical particular-solution rule; note the
    recorded caveat — on a rank-deficient system the zeroed particular
    solution may fail a caller's integrality/sign postcondition even
    though another point of the affine solution set satisfies it, so a
    caller's rejection certifies only this canonical representative).

    Returns `(sol, pivots)` — `sol` a list of `Fraction`s (free entries
    zero), `pivots` the pivot column indices — or `None` if the system
    is inconsistent.  Per-site postconditions (integrality, sign
    windows, Laurent columns, verify-by-substitution) stay with the
    callers.
    """
    from fractions import Fraction
    m = len(cols)
    n = dim if dim is not None else len(rhs)
    aug = [[Fraction(cols[j][i]) for j in range(m)] + [Fraction(rhs[i])]
           for i in range(n)]
    piv = []
    r = 0
    for c in range(m):
        p = next((i for i in range(r, n) if aug[i][c] != 0), None)
        if p is None:
            continue
        aug[r], aug[p] = aug[p], aug[r]
        inv = aug[r][c]
        aug[r] = [x / inv for x in aug[r]]
        for i in range(n):
            if i != r and aug[i][c] != 0:
                f = aug[i][c]
                aug[i] = [a - f * b for a, b in zip(aug[i], aug[r])]
        piv.append(c)
        r += 1
    if any(all(x == 0 for x in row[:-1]) and row[-1] != 0 for row in aug):
        return None
    sol = [Fraction(0)] * m
    for i, c in enumerate(piv):
        sol[c] = aug[i][-1]
    return sol, piv


class SkeinConeData(ConeData):
    """`ConeData` generated from a surface's simple-curve ray dictionary
    plus a certified skein engine.  See the module docstring for the
    geometric dictionary.

    Subclasses supply, beside the usual `ConeData` bijection
    (`to_cone_label` / `from_cone_label`) and predicates (`q_commute` /
    `cocycle`):

      * `ray_kind(g)` — ``"char"`` for Wilson-line / closed-simple-curve
        rays (SU(2)-character generators), ``"monomial"`` for open-arc
        rays;
      * `cone_sets()` — the covering compatible (disjoint-ray) subsets;
        may be a lazy, unbounded generator (subclasses with infinitely
        many cones must also override `cone_of_label` with a direct
        construction — the label knows its own rays);
      * `engine_product(g, h)` — the certified engine product
        `L_g · L_h` as an `Element` over native labels, called only for
        non-q-commuting (crossing) ray pairs.  The engine carries the
        standing guard.
    """

    # -- new abstract surface ----------------------------------------------

    @abstractmethod
    def ray_kind(self, g: Label) -> str:
        """``"char"`` (Wilson / closed simple curve), ``"monomial"``
        (open arc), or ``"torus"`` for an *adjoined invertible* ray
        generator (the formal-monodromy μ — a ray generator, NOT an
        ordinary skein; the taxonomy's
        `torus_gens`).  Torus rays come in inverse pairs declared via
        `_torus_inverse_letter`; the cancellation `μ·μ⁻¹ = 1` is an
        engine cross-product (the pair does not q-commute)."""

    @abstractmethod
    def cone_sets(self) -> Iterable[frozenset]:
        """The covering compatible subsets of rays (disjoint families).
        Lazy / unbounded generators are allowed."""

    @abstractmethod
    def engine_product(self, g: Label, h: Label) -> Element:
        """`L_g · L_h` from the certified skein engine (crossing pairs
        only).  The standing guard lives here."""

    # -- derived: cones with the char partition -----------------------------

    def _char_part(self, gens: frozenset) -> frozenset:
        return frozenset(g for g in gens if self.ray_kind(g) == "char")

    def _torus_part(self, gens: frozenset) -> frozenset:
        return frozenset(g for g in gens if self.ray_kind(g) == "torus")

    def _make_cone(self, gens: frozenset) -> Cone:
        gens = frozenset(gens)
        return Cone(self, gens, torus_gens=self._torus_part(gens),
                    char_gens=self._char_part(gens))

    def iter_cones(self):
        for gens in self.cone_sets():
            yield self._make_cone(gens)

    def _cone_for_gens(self, gens_fs: frozenset) -> Cone:
        """The covering cone containing `gens_fs`.  Default: scan
        `iter_cones()` (fine for finite families); infinite-family
        subclasses override `cone_of_label` (and may override this)
        with a direct construction."""
        for c in self.iter_cones():
            if gens_fs <= c.mult_gens():
                return c
        raise ValueError(
            f"SkeinConeData: no covering cone contains {set(gens_fs)}"
        )

    def cone_of_label(self, native_label: Label) -> Cone:
        gens_fs, _ = self.to_cone_label(native_label)
        if not gens_fs:
            for c in self.iter_cones():
                return c
            raise ValueError("cone_of_label: no cones available")
        return self._cone_for_gens(gens_fs)

    # -- derived: char-aware multiply ----------------------------------------
    #
    # The generic `ConeData.derived_multiply` is char-blind (it treats a
    # literal word `χ₁^n` in a character cone as the canonical label at
    # power n — the recorded tier gap).  The skein tier makes the
    # char-aware route the default, generalizing the blessed
    # `SU2A1D3GammaConeData` pattern: expand inputs canonical → literal
    # words (`element_to_words` — Chebyshev on char rays, q-phase on
    # monomial rays), reduce with the base word machinery, and fold
    # char-cone leaves back through `Cone.pbw_to_canonical`.

    def derived_multiply(self, a: Label, b: Label) -> Element:
        if getattr(self, "_reduce_word_cache", None) is None:
            self._reduce_word_cache = {}
        one = self._q_one(0)
        words_a = self.element_to_words(Element({a: one}))
        words_b = self.element_to_words(Element({b: one}))
        acc: dict = {}
        for ca, wa in words_a:
            for cb, wb in words_b:
                for lbl, c in self._reduce_word(wa + wb).items():
                    add = ca * cb * c
                    acc[lbl] = (acc[lbl] + add) if lbl in acc else add
        return Element({l: c for l, c in acc.items() if not c.is_zero()})

    def _assert_char_central(self, chars: frozenset, gens) -> None:
        """The char routes treat character letters as phase-free against
        the rest of the label/word — valid iff each char gen has
        vanishing cocycle against every *other gen actually present*
        (not the whole cone).  True for the skein rulings' char rays
        (central Wilsons; singleton char cones)."""
        for c in chars:
            for g in gens:
                if g != c:
                    assert self.cocycle(c, g) == 0, (
                        "char route needs the character ray to commute "
                        f"with the label's gens; cocycle({c}, {g}) != 0"
                    )

    def _reduction_step(self, w: tuple):
        """Char-aware leaf: a single-cone word containing a character
        ray folds through `pbw_to_canonical` (the literal `χ₁^n` is the
        Chebyshev sum of canonicals) — with the **monomial-part phase**
        restored: `pbw_to_canonical` identifies words with canonicals
        phase-free, while the true relation is `L_{(ℓ,P)} =
        q^{φ(P)}·χ_ℓ·M` (φ = the universal `cone_label_phase`, to which
        char letters contribute 0), so the literal word folds with
        `q^{−φ}`.  Char gens must commute with the word's other letters
        (asserted).  All other words fall through to the base reducer
        step."""
        cone_fs = self._word_cone(w)
        if cone_fs is not None and any(
            self.ray_kind(g) == "char" for g in cone_fs
        ):
            chars = frozenset(g for g in cone_fs
                              if self.ray_kind(g) == "char")
            self._assert_char_central(chars, cone_fs)
            cone = self._cone_for_gens(cone_fs)
            c2, sorted_w = self._sort_within_cone(self._q_one(0), w, cone_fs)
            gens, powers = self._word_to_gens_powers(sorted_w)
            phase = self.cone_label_phase(gens, powers)
            el = cone.pbw_to_canonical(dict(powers))
            scale = c2 * self._q_one(-phase)
            return ("leaf", {lbl: scale * c for lbl, c in el.terms.items()})
        return super()._reduction_step(w)

    # -- derived: engine-backed cross products ------------------------------

    def cross_product(self, g: Label, h: Label) -> Sequence[CrossProductTerm]:
        """`L_g L_h` for crossing rays, via the certified engine, as
        literal cone words (see the module docstring for the phase
        convention)."""
        return self.element_to_words(self.engine_product(g, h))

    def element_to_words(self, el: Element) -> Sequence[CrossProductTerm]:
        """Re-express an `Element` over native canonical labels as
        `(coeff, literal word)` summands consumable by the reducer.

        Monomial-cone labels: one word per label, coefficient shifted by
        `q^{+cone_label_phase}` (the reducer folds the word back with
        `q^{-phase}`).  Character-cone labels: the Chebyshev
        `canonical_to_pbw` expansion (literal words; requires the char
        cone's internal cocycles to vanish — asserted)."""
        out: list[CrossProductTerm] = []
        for lbl, coeff in el.terms.items():
            gens_fs, powers = self.to_cone_label(lbl)
            if not gens_fs:
                out.append((coeff, ()))
                continue
            cone = self._cone_for_gens(gens_fs)
            chars = cone.char_gens() & gens_fs
            # The universal convention phase (char letters contribute 0,
            # so this is the monomial-part phase φ(P) in the char route).
            phase = self.cone_label_phase(gens_fs, powers)
            if chars:
                # Character route: L_{(ℓ,P)} = q^{φ(P)}·χ_ℓ·M with χ_ℓ
                # expanded to literal words by `canonical_to_pbw`
                # (phase-free on the char letters — asserted central).
                self._assert_char_central(chars, gens_fs)
                for word, c in cone.canonical_to_pbw(lbl).items():
                    out.append((coeff * c * self._q_one(phase), word))
                continue
            order = self.canonical_cone_order(gens_fs)
            word: tuple = ()
            for g in order:
                word = word + (g,) * powers.get(g, 0)
            out.append((coeff * self._q_one(phase), word))
        return tuple(out)


class IntrinsicBridgeSkeinConeData(SkeinConeData):
    """Absorb a certified per-chart skein BRIDGE as `SkeinConeData`.

    The bridge classes (`SkeinHeptagonKAlg`, `SkeinNonagonKAlg`,
    `SkeinSquareKAlg`, `SkeinU1HexagonKAlg`, …) compute `multiply`
    genuinely skein-side (pinned/stated engines) with every coefficient
    asserted against the intrinsic realisation — the standing guard.
    This adapter re-presents such a bridge on the cone tier:

      * the **combinatorics** (cones, label bijection, cocycle, order,
        phases, canonicalisation, torus pairs) delegate to the
        *intrinsic* realisation's certified `ConeData` — the chart data
        the skein side measured equal (`σ_Δ = B`);
      * the **cross-products** come from the bridge engine
        (`engine_product` = the bridge's guarded multiply);
      * `ray_kind` is derived: letters with an intrinsic torus-inverse
        are ``"torus"``, letters named in `char_rays` (or tagged
        `char_gens` by the intrinsic cones) are ``"char"``, the rest are
        ``"monomial"`` open chords.
    """

    def __init__(self, intrinsic_cone_data, engine_product_fn,
                 char_rays: frozenset = frozenset()):
        self._int = intrinsic_cone_data
        self._engine_fn = engine_product_fn
        chars = set(char_rays)
        # inherit char tags from the intrinsic partitions (finite scan)
        try:
            for c in self._int.iter_cones():
                chars |= set(c.char_gens())
        except NotImplementedError:
            pass
        self._char = frozenset(chars)
        self.bilateral_layer1 = getattr(
            intrinsic_cone_data, "bilateral_layer1", False)

    # -- SkeinConeData surface ------------------------------------------------

    def ray_kind(self, g) -> str:
        if g in self._char:
            return "char"
        if self._int._torus_inverse_letter(g) is not None:
            return "torus"
        return "monomial"

    def cone_sets(self):
        for c in self._int.iter_cones():
            yield c.mult_gens()

    def mult_gens(self):
        """Delegated global ray set (the intrinsics are
        `FiniteConeData`; object-layer sample harvesting reads this)."""
        return self._int.mult_gens()

    def engine_product(self, g, h) -> Element:
        """The bridge engines multiply NATIVE labels, while the reducer
        hands us cone LETTERS (abstract ids — `(a, i)` chord codes,
        `E_GEN`, …): convert each letter to its native single-generator
        label first (`from_cone_label` on the singleton — phase-free by
        the universal convention), exactly as the bridge classes do
        internally."""
        g_native = self._int.from_cone_label(frozenset({g}), {g: 1})
        h_native = self._int.from_cone_label(frozenset({h}), {h: 1})
        return self._engine_fn(g_native, h_native)

    # -- keep the intrinsic Cone partitions (order/phase source of truth) -----

    def iter_cones(self):
        return self._int.iter_cones()

    def cone_of_label(self, native_label):
        return self._int.cone_of_label(native_label)

    def _cone_for_gens(self, gens_fs):
        for c in self._int.iter_cones():
            if gens_fs <= c.mult_gens():
                return c
        raise ValueError(
            f"IntrinsicBridgeSkeinConeData: no covering cone contains "
            f"{set(gens_fs)}"
        )

    # -- straight delegation of the certified combinatorics --------------------

    def coefficient_ring(self):
        return self._int.coefficient_ring()

    def to_cone_label(self, native_label):
        return self._int.to_cone_label(native_label)

    def from_cone_label(self, gens, powers):
        return self._int.from_cone_label(gens, powers)

    def q_commute(self, g, h) -> bool:
        return self._int.q_commute(g, h)

    def cocycle(self, g, h) -> int:
        return self._int.cocycle(g, h)

    def canonical_cone_order(self, gens):
        return self._int.canonical_cone_order(gens)

    def cone_label_phase(self, gens, powers) -> int:
        return self._int.cone_label_phase(gens, powers)

    def canonicalize_cone_label(self, cone, gens, powers):
        return self._int.canonicalize_cone_label(cone, gens, powers)

    def _torus_inverse_letter(self, g):
        return self._int._torus_inverse_letter(g)

    def cycle_period_bound(self) -> int:
        return self._int.cycle_period_bound()
