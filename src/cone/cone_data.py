"""Cone-filtration data for K_𝖖-algebras.

Many K_𝖖-algebras carry a *cone-filtration* structure: the canonical basis
splits into "cones" — maximal collections of pairwise q-commuting basis
elements — and within each cone the basis is a q-normal-ordered
monomial in a finite set of "multiplicative generators."  This module
contracts that structure as an optional sidecar to `KAlgebra` so that
subclasses can share generic algorithms (multiplication, ρ²-cycle-out
trace simplification, verifiers) instead of reimplementing them.

Two senses of "cone" — do not conflate
--------------------------------------
The word *cone* names two different convex cones in the charge lattice
`Γ`; this module is about the **second**:

  * **Positive cone** `Γ₊ ⊂ Γ` — global, fixed; the pointedness /
    positivity cone of the lattice, used for F-basis minima.  Code:
    `BPSKAlgebra.cone_gens` / `cone_witness`.  *Not this module.*
  * **Monomial cone** (this module's "cone") — chart-*local*: the
    charges whose `F_γ` is a pure cluster monomial in a given chart,
    equivalently a maximal pairwise-q-commuting (PBW) block of the
    canonical basis.  Organised here as `ConeData`'s cones.

They are related (the cluster fan of monomial cones tiles a
neighbourhood of `Γ₊`) but distinct objects.  In prose/docstrings
always qualify — "positive cone" vs "monomial cone" — never bare "cone".

The `Cone` class factors the *change of basis* between PBW monomials in
mult-gens and the
canonical basis of A_q[T] inside each cone.  `Cone` is **one concrete
class parameterized by a partition of its mult-gens** into orthogonal,
composable kinds:

  * **monomial** gens (the default / remainder) — the change of basis is
    the identity (the PBW monomial IS the canonical-basis element, modulo
    the universal `cone_label_phase`).  Pentagon, Heptagon, U1Hexagon,
    A1A2k, and similar unflavoured (or U(1)-flavour-folded-into-labels)
    closed-form theories.

  * **torus** gens — invertible directions (`v · v_inv = 1`), so the cone
    subalgebra carries a quantum-torus factor; distinct ρ-semantics (ρ can
    shear the torus directions).  **In use**: `U1A1AoddKAlg(k)` (its
    `iter_cones` yields torus cones).

  * **character** gens — the change of basis is a Chebyshev /
    Weyl-character relation `χ_k = U_k(χ_1)`: the canonical basis on a
    marked `char_gen` IS the SU(2) irreducible characters `{χ_k}` (a
    fundamental `χ_1`).  In use: the pure-SU(2) gauge-Wilson direction
    (`pure_su2_h_wilson`).  The natural home for any SU(2)-Wilson-line /
    character-ray cone (e.g. the skein Wilson lines).

The three kinds are **partitions of the one `Cone` class**, not distinct
types: a *monomial* cone has no torus/char gens, a *quantum-torus* cone
marks a `torus_gens` subset, a *character* cone marks a `char_gens` subset,
and the kinds compose orthogonally, so **mixed cones** (e.g. torus +
character together) are constructible directly via
`Cone(parent, mult_gens, torus_gens=…, char_gens=…)`.  The partition is
carried as data and queried with `is_monomial()` / `is_quantum_torus()` /
`is_character()`.  (The former thin partition-fixing subclasses
`MonomialCone` / `QTCone` / `CharacterCone` were folded into the single
parameterized class.)

The companion abstract `ConeKAlgebra` (in `cone_kalgebra.py`) is the
"closed-form" presentation tier: a `KAlgebra` defined by a `ConeData`
instance + a Layer-2 residual-trace rule.

The structure (recap)
---------------------

A `ConeData` for a `KAlgebra` `A` exposes:

  * **Multiplicative generators**: a (possibly infinite) collection of
    canonical-basis labels `g` such that the corresponding L_g generate
    the whole algebra as q-normal-ordered monomials.
  * **q-commute predicate**: `q_commute(g, h)` says whether
    `L_g L_h = q^{2 c(g, h)} L_h L_g`.  Reflexive, symmetric, NOT
    transitive in general.
  * **Cocycle**: `cocycle(g, h)` returns the integer `c(g, h)` when
    `q_commute(g, h)` (antisymmetric).
  * **Cross-product**: `cross_product(g, h)` returns `L_g L_h` as a sum
    of words in mult-gens, when `g` and `h` do NOT q-commute.
  * **Cone-label bijection**: `to_cone_label(native_label) → (gens, powers)`
    and `from_cone_label(gens, powers) → native_label`, where `gens` is
    a frozenset of pairwise q-commuting mult-gens (the canonical-basis
    element's "compatible set") and `powers` is a dict mapping each gen
    to its strictly-positive power.  Forward is easy; backward is the
    "difficult combinatorics" for BPS-like algebras.
  * **Canonical cone ordering**: `canonical_cone_order(gens) → tuple`
    fixes a linear order on the gens so the word
    `g_0^{powers[g_0]} · g_1^{powers[g_1]} · …` equals
    `L_{from_cone_label(gens, powers)}` literally (no implicit q-phase).
    Default: `tuple(sorted(gens))`.

The generic `derived_multiply(a, b)` reduces the concatenated word
`to_cone_label(a) ++ to_cone_label(b)` by (i) sliding q-commuting letters
past each other (paying cocycle phases), (ii) substituting
`cross_product` at non-q-commuting collisions, (iii) terminating when
each surviving word lies in a single cone, then reading off the native
labels via `from_cone_label`.  The algorithm is validated against the
pentagon algebra's full ordered-pair multiplication table.

Flavour in the cone presentation
--------------------------------

How the flavour symmetry `G_f` interacts with the cone structure is the
most confusing aspect of this tier: every subclass looks the same on the
surface (a `ConeData` + cross-products), yet flavour can live in **four
different places**.  Always check `coefficient_ring()` *and*
`_label_section_decompose` first.  Five patterns occur:

  **I. Unflavoured / U(1) folded into the labels** (`R =
  TrivialZPlusRing`).  The U(1) gauge/flavour torus direction is a
  *lattice/label* coordinate (a torus gen, or an exponent in
  the cone label) — there is no R-side content.
  `_label_section_decompose(label) = (label, R.one())`.  *Examples:*
  `FinitePentagonKAlgebra`, `U1SquareKAlg`, `U1OctagonKAlg`,
  `U1DecagonKAlg`, `U1A1AoddKAlg`, `U1HexagonKAlg`.  ⚠ In `U1HexagonKAlg` the central `E`
  exponent is a **dynamical algebra generator, not a flavour fugacity**
  — so `R` stays `Trivial` even though the label carries a μ-looking
  integer.

  **II. Abelian flavour in the coefficients** (`R = AbelianZPlusRing`).
  Cone labels are gauge-only (a section); the μ-fugacity lives in the
  `RLaurent` coefficients of `cross_product`
  (`RLaurent(R, {q_exp: RElement(R, {(μ,): c})})`).
  `_label_section_decompose(label) = (gauge_section,
  R.basis_element((μ,)))`.  *Examples:* `SU2Nf1KAlgebra`,
  `FiniteA3/A5/A7/E7KAlgebra`.

  **III. Non-abelian character folded into a label component** (`R =
  SU2 / SO3 / SU3ZPlusRing`).  A χ-index is a dedicated coordinate of
  the label; `_label_section_decompose` peels it onto R:
  `(tile, a, b, k) ↦ ((tile, a, b, 0), R.basis_element(k))`
  (SU(3): `(…, p, q) ↦ ((…, 0, 0), χ_{(p,q)})`).  *Examples
  (hand-written):* `A1D3KAlg`, the `A1Dodd` family (`A1D5ConeKAlg`,
  `A1D7ConeKAlg`), `SU3ADKAlg`, `FiniteA1D3/5/7KAlgebra`.

  **IV. Character cone** (a `Cone` with non-empty `char_gens`).  The
  canonical basis *is* the irreducible characters `{χ_k}` of an SU(2) acting on a
  marked subset of the mult-gens (`char_gens`); the change of variables on each
  char-gen is the SU(2) Chebyshev/Weyl relation `χ_k = U_k(χ_1)`
  (`su2_char_to_monomials`) and its inverse `χ_1^n = Σ(…)χ_j`
  (`su2_monomial_to_chars`), with the remaining gens monomial.  A subclass routes
  `derived_multiply` through `canonical_to_pbw → monomial reduce → pbw_to_canonical`
  so the Wilson Clebsch–Gordan (`χ_1·χ_1 = χ_0+χ_2`) is genuinely cone-derived
  (no hard-coding, no BPS).  *Example:* the pure-SU(2) gauge-Wilson direction
  (`pure_su2_h_wilson`).

  **V. Mixed SU(2)×U(1)** (`R = SU2xU1ZPlusRing`).  Like II, **both** the
  χ *and* the μ live in the `RLaurent` cross-product coefficients
  (`RElement` keys `(χ_k, μ_m)`); the cone label is gauge-only and
  `_label_section_decompose(label) = (label, R.one())`.  *Examples:*
  `FiniteA1D4/A1D6/A1D8KAlgebra`.

Pitfalls
~~~~~~~~

  * **Four flavour locations, one surface.**  A `ConeKAlgebra` gives no
    hint of *where* its flavour lives — the lattice/labels (I), the
    coefficients (II, V), a split-out label component (III), or a
    bypassed sector (IV).

  * **The same symmetry, opposite mechanisms.**  The SU(2) flavour of
    the *hand-written* `A1D3KAlg` lives **in the label** (III; section-
    decompose peels it); the SU(2) flavour of the *generated*
    `FiniteA1D4KAlgebra` lives **in the cross-product coefficients** (V;
    label gauge-only).  Two SU(2)-flavoured cone algebras, mirror-image
    encodings.

  * **`RLaurent` coefficients are not just q-powers.**  In II/V the
    cross-product coefficient is the full R-multiplier `RLaurent(R, {q:
    RElement(R, {char_key: c})})`; the character key is *not* part of
    the label.

  * **Character cones are implemented** (SU(2) Chebyshev).  III still realises
    the *flavour-in-a-label-slot* character idea informally (cheaper when no CG is
    needed); IV is the genuine cone-derived character multiply.

  * **Naming collision — `Hexagon` ≠ `U1Hexagon`.**  The U(1)-*flavoured*
    `(n+3)`-gons `Hexagon` / `Octagon` / `Decagon` (= `FiniteA3` /
    `FiniteA5` / `FiniteA7KAlgebra`, `AbelianZPlusRing`, Pattern II) are
    *different algebras* from the U(1)-*gauged, unflavoured*
    `U1Hexagon` / `U1Octagon` / `U1Decagon` (`u1_*_kalg.py`, `Trivial`,
    Pattern I).  Same polygon name, opposite flavour status.

  * **A μ-looking label exponent may not be flavour** (`U1Hexagon`'s
    `E`) — check `coefficient_ring()`.

The lift coordinate
~~~~~~~~~~~~~~~~~~~

The preferred surface for the section split is `KAlgebra.r_label_decompose`
— the single-irrep `(section, R-basis-label)` flavour-lift coordinate.
Across the five patterns above `_label_section_decompose` only ever takes
**two shapes**: the *trivial lift* `(label, R.one())` (gauge-only /
flavour-in-coefficients — patterns I, II-generated, V) and the
*peel-a-slot* `((…, 0), R.basis_element(slot))` (flavour weight in a label
coordinate — pattern III, and the hand-written pattern-II `SU2Nf1`).
`ConeKAlgebra` **defaults `r_label_decompose`**, reading the single irrep
off whichever shape a subclass supplies (a *reverse* bridge — see
`ConeKAlgebra.r_label_decompose`), so no cone theory needs a per-theory
edit.  New cone theories supply `r_label_decompose` directly and skip
`_label_section_decompose`; its **inverse** `r_label_compose`
(`(section, R-basis-label) → label`) is the label-producing partner (used
by `forget` / `lower_flavour` / `KAlgebraIso.from_section_map`).
`embed_R` is the central embedding `ι : R ↪ A_𝖖` (with its faithfulness /
ρ-compatibility axioms), the default mechanism behind `r_label_compose`,
and the backing of `from_R_form`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Hashable, Iterable, Sequence

from kalgebra import Element, Label
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing, ZPlusRing, RElement, RLaurent


__all__ = [
    "ConeData",
    "FiniteConeData",
    "Cone",
    "ConePolynomial",
    "CrossProductTerm",
    "ConeLabel",
]


# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------

ConeLabel = tuple[frozenset[Label], dict[Label, int]]
"""(compatible_set, strictly-positive powers).

`compatible_set` is a frozenset of pairwise q-commuting multiplicative
generators (NOT required to be maximal — just compatible).  `powers` is
a dict whose keys are exactly `compatible_set` and whose values are
integers ≥ 1.  The pair uniquely labels a canonical-basis element."""

CrossProductTerm = tuple["LaurentPoly | RLaurent", tuple[Label, ...]]
"""A single (coeff, word) summand in a cross-product expansion.

The coefficient is a `LaurentPoly` when the parent algebra's coefficient
ring is `TrivialZPlusRing()` (the default), or an `RLaurent[R]` when the
parent algebra is defined over a non-trivial Z+-ring R (Abelian or
non-Abelian character rings).  Subclasses building coefficients should
prefer `self._q_one(e)` / `self._q_zero()` over direct construction, so
that the type tracks the active R automatically.
"""


# ----------------------------------------------------------------------
# Base ABC
# ----------------------------------------------------------------------


class ConeData(ABC):
    """Optional cone-filtration structure attached to a `KAlgebra`.

    Here "cone" = **monomial cone** (a maximal pairwise-q-commuting PBW
    block), *not* the positive cone `Γ₊` of `BPSKAlgebra.cone_gens` —
    see the module docstring's "Two senses of cone".

    Concrete subclasses must supply the bijection
    (`to_cone_label`, `from_cone_label`), the q-commute predicate and
    cocycle (`q_commute`, `cocycle`), the cross-product table
    (`cross_product`), and the canonical cone ordering
    (`canonical_cone_order`).  The generic `derived_multiply` is then
    available for free.

    Coefficient ring
    ----------------
    `ConeData` defaults to operating over `TrivialZPlusRing()`
    (Z[q, q⁻¹] coefficients), which is the right ring for every
    closed-form algebra currently in the repo (Pentagon, Heptagon,
    A1A2k, U1Square, U1Hexagon, U1Octagon, U1Decagon, U1A1Aodd(k)).
    Subclasses backing a `KAlgebra` over a non-trivial Z+-ring R
    (Abelian fugacity ring `AbelianZPlusRing`, or non-Abelian
    character rings `SU2ZPlusRing` / `SO3ZPlusRing` / `SU3ZPlusRing`)
    override `coefficient_ring()` to return that R; the framework
    primitives then construct `RLaurent[R]` coefficients in place of
    `LaurentPoly`, and `cross_product` subclass implementations are
    expected to do the same (via the `_q_one(e)` / `_q_zero()`
    helpers below, or by constructing `RLaurent(R, ...)` directly).

    Layer-1 reduction direction
    ---------------------------
    `bilateral_layer1` (default `False`) selects the tagged-cyclicity
    search strategy in `_tagged_cyclicity_round`.  When `False` the
    reducer runs Form B (ρ⁻²) to its first Plücker and only falls back to
    Form A (ρ²) if B exhausts the cycle bound.  When `True` it *races*
    both forms and uses whichever fires a Plücker in fewer cycles ("which
    Plücker first") — a pure performance heuristic: both forms reduce a
    trace monomial to the *same* seed-`Element` (the trace is
    well-defined), so the choice never changes the answer, only the size
    of the reduction tree.  See `_tagged_cyclicity_round_bilateral`.
    """

    bilateral_layer1: bool = False

    # -- coefficient ring + qpoly factory ---------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        """The Z+-ring R over which the parent K_𝖖-algebra is defined.
        Default `TrivialZPlusRing()` ⇒ Z[q, q⁻¹] coefficients
        (`LaurentPoly` everywhere in the cone pipeline).  Subclasses
        backing a flavoured / character-ring K_𝖖-algebra override.

        When wired into a `ConeKAlgebra`, must agree with
        `parent.coefficient_ring()`.
        """
        return TrivialZPlusRing()

    def _q_one(self, e: int) -> "LaurentPoly | RLaurent":
        """The unit q-monomial `q^e · 1_R`, typed appropriately for
        the coefficient ring.  `LaurentPoly({e: 1})` for trivial R
        (preserves byte-identical behaviour for the existing closed-
        form algebras); `RLaurent(R, {e: 1})` for non-trivial R."""
        R = self.coefficient_ring()
        if isinstance(R, TrivialZPlusRing):
            return LaurentPoly({e: 1})
        return RLaurent(R, {e: 1})

    def _q_zero(self) -> "LaurentPoly | RLaurent":
        """The zero element of the coefficient ring (typed)."""
        R = self.coefficient_ring()
        if isinstance(R, TrivialZPlusRing):
            return LaurentPoly()
        return RLaurent(R)

    def _rho2_twist_unit(self, alg, tag_native, sign: int):
        """Central-unit correction for ELEMENT-level `ρ^{±2}` on the
        single-mult-gen label `tag_native`.  The Layer-1 reducer must
        account for the central μ^δ shift here, or the ρ²-cyclicity
        slide would silently drop a unit on flavoured entries.

        ρ inverts central flavour characters (`ρ(μ·x) = μ⁻¹·ρ(x)`),
        and on unit-character entries each letter carries a μ-shift
        under ρ (`ρ(L_w) = μ^{δ(w)}·L_{Pw}`, the algebra's
        `_rho_delta` table, validated operationally by exhaustive
        ρ-automorphism sweeps).  Composing:

            ρ²(L_t)  = μ^{δ(Pt) − δ(t)}   · L_{P²t}
            ρ⁻²(L_u) = μ^{δ(P⁻²u) − δ(P⁻¹u)} · L_{P⁻²u}

        Returns the μ-unit as a coefficient multiplier (`RLaurent` at
        q⁰), or `None` when the algebra carries no δ-table or the net
        shift vanishes — in which case callers skip the multiply
        (on trivial/su2 entries label-ρ IS element-ρ and no correction
        arises)."""
        delta = getattr(alg, "_rho_delta", None)
        if not delta:
            return None

        def dword(native):
            gens, powers = self.to_cone_label(native)
            tot = None
            for g, p in powers.items():
                di = delta.get(g)
                if di is None:
                    continue
                if tot is None:
                    tot = [0] * len(di)
                for j in range(len(di)):
                    tot[j] += di[j] * p
            return tot

        if sign > 0:
            mid = alg.rho(tag_native)
            d_from, d_to = dword(tag_native), dword(mid)
        else:
            mid = alg.rho_inverse(tag_native)
            fin = alg.rho_inverse(mid)
            d_from, d_to = dword(mid), dword(fin)
        n = len(d_from or d_to or ())
        net = tuple((0 if d_to is None else d_to[j])
                    - (0 if d_from is None else d_from[j])
                    for j in range(n))
        if not any(net):
            return None
        R = self.coefficient_ring()
        # `net` is the U(1)/central shift in the Λ (1-dim-rep) lattice — a
        # rank-1 tuple from `_rho_delta`.  It must be embedded into R's basis
        # via `embed_one_dim_rep`, NOT used as a raw basis key: for a pure-U(1)
        # ring (AbelianZPlusRing) the embedding is the identity (so frozen u1
        # entries stay byte-identical), but for a product ring like
        # SU2xU1ZPlusRing the basis key is `(0, m)` (trivial SU(2) ⊗ μ^m), and
        # the raw 1-tuple `(m,)` is malformed (the a1d8 deep-word reducer
        # crash: `multiply_basis` got a 1-tuple key).
        key = R.embed_one_dim_rep(net)
        return RLaurent(R, {0: RElement(R, {key: 1})})

    # -- subclass-supplied primitives -------------------------------------

    @abstractmethod
    def to_cone_label(self, native_label: Label) -> ConeLabel:
        """Native canonical-basis label → (compatible_set, powers)."""

    @abstractmethod
    def from_cone_label(
        self, gens: frozenset[Label], powers: dict[Label, int]
    ) -> Label:
        """Inverse of `to_cone_label`.  Returns the native label of the
        canonical-basis element `L_{∏ g^{powers[g]}}` for `g ∈ gens`.

        Must satisfy: `from_cone_label(*to_cone_label(x)) == x`.
        """

    @abstractmethod
    def q_commute(self, g: Label, h: Label) -> bool:
        """True iff `L_g L_h = q^{2c} L_h L_g` for some `c ∈ ℤ`."""

    @abstractmethod
    def cocycle(self, g: Label, h: Label) -> int:
        """The integer `c(g, h)` with `L_g L_h = q^{2c(g,h)} L_h L_g`.

        Antisymmetric: `cocycle(g, h) = -cocycle(h, g)`.  Defined only
        when `q_commute(g, h)`; otherwise behaviour is undefined.
        """

    @abstractmethod
    def cross_product(
        self, g: Label, h: Label
    ) -> Sequence[CrossProductTerm]:
        """`L_g L_h` as a sequence of `(coeff, word)` summands.

        Defined only when NOT `q_commute(g, h)`.  Each summand is a
        LaurentPoly coefficient times a word in mult-gens (possibly
        empty = identity term).
        """

    def canonical_cone_order(
        self, gens: frozenset[Label]
    ) -> tuple[Label, ...]:
        """Canonical linear order on the mult-gens of a cone.

        The word `g_0^{powers[g_0]} · g_1^{powers[g_1]} · …` in this
        order is the *literal* product of mult-gens; it relates to
        `L_{from_cone_label(gens, powers)}` by the convention phase
        returned by `cone_label_phase`.  Default: Python's `sorted`.
        Subclasses with conventions that don't match `sorted` (e.g.
        pentagon's wrap-around cone `{0, 4}` whose natural order is
        `(4, 0)`) must override.
        """
        return tuple(sorted(gens))

    def cone_label_phase(
        self, gens: frozenset[Label], powers: dict[Label, int]
    ) -> int:
        """Convention phase relating the canonical-basis element to the
        literal mult-gen product:

            `L_{from_cone_label(gens, powers)}  =  q^{cone_label_phase}
                                                   ·  ∏ L_{g}^{powers[g]}`

        where the product is taken in `canonical_cone_order(gens)`.

        **Universal formula (bar-invariance).** Across all cone-data
        structures the canonical basis is fixed by bar-invariance to

            `L_{(a_1, ..., a_n)} = q^{-Σ_{i<j} c(a_i, a_j)}
                                   · L_{a_1} · L_{a_2} · ... · L_{a_n}`

        where `(a_1, ..., a_n)` is the canonical-ordered flat word.  So
        the convention phase is just `-Σ_{i<j} c(a_i, a_j)` summed over
        all in-order pairs in that word, and is derivable from
        `cocycle` + `canonical_cone_order` alone.  Subclasses do not
        need to override.
        """
        order = self.canonical_cone_order(gens)
        word: list[Label] = []
        for g in order:
            word.extend([g] * powers.get(g, 0))
        total = 0
        for i in range(len(word)):
            for j in range(i + 1, len(word)):
                total += self.cocycle(word[i], word[j])
        return -total

    def canonicalize_cone_label(
        self,
        cone: frozenset[Label],
        gens: frozenset[Label],
        powers: dict[Label, int],
    ) -> "tuple[frozenset[Label], dict[Label, int], int]":
        """Canonicalize a cone-internal label for non-simplicial cones.

        For cones with linearly-dependent rays (e.g. the decagon's
        size-10 max cones in a 6-dim gauge space), two distinct
        ``(gens, powers)`` tuples can represent the same algebra
        element — they correspond to the same lattice point but use
        different ray decompositions.

        This method returns the **canonical** ``(gens', powers', q_phase)``
        such that

            ``L_{(gens, powers)}  =  q^{q_phase} · L_{(gens', powers')}``

        in the algebra.  ``derived_multiply`` calls this before
        forming the final native label, so two products giving the
        same algebra element collapse onto a single ``Element`` term.

        Default implementation: identity (simplicial case — no
        identifications).  Subclasses for non-simplicial cones
        override.
        """
        return gens, powers, 0

    # -- generic derived methods ------------------------------------------

    def derived_multiply(self, a: Label, b: Label) -> Element:
        """Compute `L_a · L_b` as an `Element` over native labels using
        only the cone-data primitives.

        Algorithm:
          1. Convert `a`, `b` to canonical-ordered mult-gen words.
          2. Concatenate; reduce by sliding q-commuting letters
             (cocycle-phased) and substituting `cross_product` at
             non-q-commuting collisions.
          3. For each surviving single-cone word, sort into canonical
             order (accumulating cocycle phase), read off `(gens, powers)`,
             call `from_cone_label`, and accumulate into the output
             `Element`.
        """
        gens_a, powers_a = self.to_cone_label(a)
        gens_b, powers_b = self.to_cone_label(b)
        word_a = self._cone_label_to_word(gens_a, powers_a)
        word_b = self._cone_label_to_word(gens_b, powers_b)
        # Convention phase: L_a = q^{phase_a} · word_a, similarly for b.
        # So L_a · L_b = q^{phase_a + phase_b} · (word_a · word_b).
        init_phase = (
            self.cone_label_phase(gens_a, powers_a)
            + self.cone_label_phase(gens_b, powers_b)
        )

        cache = getattr(self, "_reduce_word_cache", None)
        if cache is None:
            cache = self._reduce_word_cache = {}
        reduced = self._reduce_word(word_a + word_b)
        scale = self._q_one(init_phase)
        return Element({lbl: scale * c for lbl, c in reduced.items()
                        if not c.is_zero()})

    def _reduce_word(self, root: tuple) -> dict:
        """Memoised reduction of a bare mult-gen word to `{native: LaurentPoly}`
        (the coefficient-1 reduction; the caller scales by the convention
        phase).

        Iterative post-order over the cross-product / bubble DAG, **memoising
        every word** so each distinct sub-word is reduced exactly once.  The old
        work-queue re-reduced re-convergent words (the same monomial reached by
        many cross-product paths), which blew up exponentially and tripped a
        100k-step cap on deep products; merging those paths makes the cost
        roughly linear in the number of *distinct* sub-words (e.g. an a1d6
        degree-6 product: 100k-step timeout → ~4.5k sub-words, < 2 s).  Exact —
        no truncation; the cache is intrinsic to the cone data (shared across
        `derived_multiply` calls)."""
        cache = self._reduce_word_cache
        if root in cache:
            return cache[root]
        stack = [root]
        misses = 0
        while stack:
            w = stack[-1]
            if w in cache:
                stack.pop()
                continue
            kind, data = self._reduction_step(w)
            if kind == "leaf":
                cache[w] = {n: p for n, p in data.items() if not p.is_zero()}
                stack.pop()
                continue
            pending = [cw for (_coeff, cw) in data if cw not in cache]
            if pending:
                misses += 1
                if misses > 2_000_000:
                    raise RuntimeError(
                        f"_reduce_word: > 2e6 distinct sub-words reducing "
                        f"{root} (genuine non-termination?)")
                stack.extend(pending)
            else:
                res: dict = {}
                for (coeff, cw) in data:
                    for nat, p in cache[cw].items():
                        res[nat] = (res[nat] + coeff * p if nat in res
                                    else coeff * p)
                cache[w] = {n: p for n, p in res.items() if not p.is_zero()}
                stack.pop()
        return cache[root]

    def _reduction_step(self, w: tuple):
        """One reduction step of a mult-gen word `w`.

        Returns `("leaf", {native: poly})` if `w` lies in a single cone (sorted,
        canonicalised, phased), else `("node", [(coeff, child_word), …])` — the
        cross-product daughters at the first non-q-commuting *adjacency*, or (if
        none is adjacent) the cocycle-phased *bubble* that exposes one.  This is
        the per-step kernel the memoised `_reduce_word` drives; it mirrors the
        former inline work-queue exactly."""
        cone = self._word_cone(w)
        if cone is not None:
            c2, sorted_w = self._sort_within_cone(self._q_one(0), w, cone)
            gens, powers = self._word_to_gens_powers(sorted_w)
            gens, powers, canon_phase = self.canonicalize_cone_label(
                cone, gens, powers,
            )
            native = self.from_cone_label(gens, powers)
            phase_out = self.cone_label_phase(gens, powers)
            return ("leaf", {native: c2 * self._q_one(canon_phase - phase_out)})

        # Multi-cone: fire at the first non-q-commuting adjacency.
        for k in range(len(w) - 1):
            g, h = w[k], w[k + 1]
            if g == h:
                continue
            if not self.q_commute(g, h):
                return ("node", [(coeff, w[:k] + replacement + w[k + 2:])
                                 for (coeff, replacement)
                                 in self.cross_product(g, h)])

        # No adjacent collision: bubble a non-q-commuting pair together.
        target_i = target_j = None
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                if not self.q_commute(w[i], w[j]):
                    target_i, target_j = i, j
                    break
            if target_i is not None:
                break
        if target_i is None:
            # Defensive (shouldn't happen: _word_cone returned None).
            gens, powers = self._word_to_gens_powers(w)
            native = self.from_cone_label(gens, powers)
            phase_out = self.cone_label_phase(gens, powers)
            return ("leaf", {native: self._q_one(-phase_out)})
        new_w = list(w)
        phase_exp = 0
        for k in range(target_j, target_i, -1):
            partner = new_w[k - 1]
            tagged = new_w[k]
            if not self.q_commute(partner, tagged):
                cur = tuple(new_w)
                return ("node", [
                    (coeff * self._q_one(phase_exp),
                     cur[:k - 1] + replacement + cur[k + 1:])
                    for (coeff, replacement)
                    in self.cross_product(partner, tagged)])
            phase_exp += 2 * self.cocycle(partner, tagged)
            new_w[k - 1], new_w[k] = new_w[k], new_w[k - 1]
        # Tagged letter now adjacent to w[target_i]; re-reduce the bubbled word.
        return ("node", [(self._q_one(phase_exp), tuple(new_w))])

    # -- ρ²-cycle-out trace simplification --------------------------------

    def simplify_trace_via_cone_data(
        self, alg: "KAlgebra", native_label: Label
    ) -> Element:
        """Reduce `L_{native_label}` to an `Element` whose support is
        the algebra's *trace seeds* (= identity + single mult-gens), with
        the same trace as the input.

        Scope — **applicable only to monomial-cone ConeData**
        (Pentagon, Heptagon, U1Hexagon, A1A2k, …).  The tagged-cyclicity
        algorithm relies on the following structural property: for the
        tagged mult-gen `g_1` and the remaining factors `g_2, …, g_n`,
        **some power of ρ² applied to `g_1` lands outside the cone of
        the remaining factors** (= fails to q-commute with at least
        one of them), exposing a cross-cone Plücker that triggers the
        reduction.  For quantum-torus ConeData (e.g.
        `U1SquareConeData`), ρ² applied to any cone-mult-gen stays
        inside the cone of the remaining factors (it just shifts torus
        exponents within the same cone), so no Plücker ever fires;
        tagged-cyclicity is **inapplicable by design** for quantum-torus
        cones.  Trace handling for quantum-torus algebras is per-theory
        at Layer 2 (e.g. `U1SquareKAlg.trace` delegates to an SQED₁
        closed-form path; the general quantum-torus trace requires
        cyclicity recursions of the form
        `Tr(v²) = q Tr(1) + q(q−1) Tr(v)`).

        Algorithm — **tagged cyclicity**.  In an L-product word `[g_1, ...,
        g_n]` of mult-gens:

          1. Pop the front letter `g_1` (tagged), apply `ρ⁻²`, push to the
             back.  (Form B cyclicity: `Tr(L_τ · X) = Tr(X · ρ⁻²(L_τ))`.)
             This is one cycle; multiple cycles compose `ρ⁻²ᵏ` on the tag.
          2. Slide the tagged letter back toward the front, q-commuting
             past each left-partner with cocycle phase.  If the partner
             does *not* q-commute (cross-cone collision), apply
             `cross_product` at that adjacency, splice each daughter back
             into the word, and recurse.
          3. If the tag returns to the front with no collision, cycle again
             (= another `ρ⁻²` applied).
          4. Base cases.  Empty word → coefficient at the identity native
             label.  Single letter `L_g` → coefficient at the single-mult-gen
             native label `from_cone_label({g}, {g: 1})`.

        The dual "Form A" variant — `Tr(X · L_τ) = Tr(ρ²(L_τ) · X)`, popping
        the back letter and pushing to the front with `ρ²` — is also valid;
        the algorithm tries Form B first and falls back to Form A only when
        Form B fails to make progress within `cycle_period_bound` cycles.

        Returns: `Element` `e` over native labels (each summand a trace
        seed) with `alg.trace_element(e) == alg.trace(L_{native_label})`.
        """
        gens, powers = self.to_cone_label(native_label)
        # Universal early termination: if some ρ²-fixed factor has non-
        # trivial net q-commute cocycle with the rest of the monomial,
        # the trace vanishes by ρ²-cyclicity (= the Tr(v · O) = 0 rule).
        # See `trace_vanishes_by_rho2_fixed_factor`.  No-op for algebras
        # without ρ²-fixed mult-gens (Pentagon / Heptagon / A1A2k).
        if self.trace_vanishes_by_rho2_fixed_factor(alg, gens, powers):
            return Element({})
        if not gens or sum(powers.values()) <= 1:
            # Trivially a trace seed.  Round-trip through `from_cone_label`
            # so the result is in the subclass's *canonical* native form
            # (e.g. pentagon collapses `(1, 0, 0)` to `(0, 0, 0)`).  Pass
            # through the ρ²-orbit canonicalisation so the returned seed
            # is the canonical orbit representative — same contract as
            # the post-reduction path below.
            raw = Element.basis(self.from_cone_label(gens, powers))
            return self._collapse_rho2_orbits_in_element(alg, raw)

        word = self._cone_label_to_word(gens, powers)
        # Initial phase: the input native label's `cone_label_phase` —
        # the literal word is `q^{-phase_in}` times the canonical-basis
        # element, so the trace of the canonical-basis element equals
        # `q^{phase_in}` times the trace of the literal word.  We track
        # phase as a LaurentPoly factor on each work item.
        phase_in = self.cone_label_phase(gens, powers)
        cache: dict = {}
        reduced = self._simplify_word(alg, tuple(word), cache)
        scale = self._q_one(phase_in)
        raw = Element({lbl: scale * c for lbl, c in reduced.items()
                       if not c.is_zero()})
        # Layer-1 boundary: collapse ρ²-orbits on the seed support (the
        # second half of Layer 1).  See `_collapse_rho2_orbits_in_element`.
        return self._collapse_rho2_orbits_in_element(alg, raw)

    def _simplify_word(self, alg, root: tuple, cache: dict) -> dict:
        """Memoised reduction of a bare word to `{seed: poly}` (coeff 1) by
        tagged ρ²-cyclicity — the trace analogue of `_reduce_word`.

        Iterative post-order over the cyclicity/Plücker daughters, memoising
        every word so re-convergent reductions (the same word reached by many
        ρ²-cycle / cross paths) are reduced once.  This removes the work-queue
        blow-up that tripped the 200k-step cap on deep su2u1 / e8 trace seeds.
        Exact (no truncation); `cache` is per-call (the blow-up is within one
        reduction)."""
        if root in cache:
            return cache[root]
        stack = [root]
        misses = 0
        while stack:
            w = stack[-1]
            if w in cache:
                stack.pop()
                continue
            kind, data = self._simplify_step(alg, w)
            if kind == "leaf":
                cache[w] = {n: p for n, p in data.items() if not p.is_zero()}
                stack.pop()
                continue
            pending = [cw for (_coeff, cw) in data if cw not in cache]
            if pending:
                misses += 1
                if misses > 2_000_000:
                    raise RuntimeError(
                        f"_simplify_word: > 2e6 distinct sub-words reducing "
                        f"{root} (genuine non-termination?)")
                stack.extend(pending)
            else:
                res: dict = {}
                for (coeff, cw) in data:
                    for nat, p in cache[cw].items():
                        res[nat] = (res[nat] + coeff * p if nat in res
                                    else coeff * p)
                cache[w] = {n: p for n, p in res.items() if not p.is_zero()}
                stack.pop()
        return cache[root]

    def _simplify_step(self, alg, w: tuple):
        """One tagged-cyclicity step of word `w` (coefficient 1).  Returns
        `("leaf", {seed: poly})` if `w` reduces to a trace seed (empty/single
        letter, or an exhausted single-cone word), else `("node", [(coeff,
        child_word), …])` — the Plücker/cross daughters.  Mirrors the former
        inline work-queue body step-for-step."""
        one = self._q_one(0)
        if len(w) <= 1:
            if not w:
                native = self.from_cone_label(frozenset(), {})
            else:
                native = self.from_cone_label(frozenset({w[0]}), {w[0]: 1})
            return ("leaf", {native: one})
        # Canonicalise a single-cone word to its sorted/canonical representative
        # before tagged cyclicity, mirroring multiply's `_reduction_step`.  Two
        # orderings of the same cone monomial are the same algebra element (up to
        # a cocycle phase), but `_trace_residual` reads its closed form off the
        # *label*; without this redirect a non-canonical ordering reaches Layer-2
        # as a different seed and is mis-evaluated (the order-sensitivity that
        # injected a spurious `-q^14` into the a1aodd k=3 diameter-power trace,
        # caught against the BPS-free oracle).
        cone = self._word_cone(w)
        if cone is not None:
            c2, sorted_w = self._sort_within_cone(one, w, cone)
            gens_c, powers_c = self._word_to_gens_powers(sorted_w)
            gens_c, powers_c, canon_phase = self.canonicalize_cone_label(
                cone, gens_c, powers_c)
            canonical_word = tuple(self._cone_label_to_word(gens_c, powers_c))
            if canonical_word != tuple(w):
                return ("node",
                        [(c2 * self._q_one(canon_phase), canonical_word)])
        if not self.tagged_cyclicity_trigger(w):
            result = "exhausted"
        else:
            result = self._tagged_cyclicity_round(alg, one, list(w))
        if result == "exhausted":
            if self._word_cone(w) is None:
                return ("node", [(co, tuple(nw)) for (co, nw)
                                 in self._exhausted_cross_reduce(
                                     alg, one, list(w))])
            gens_w, powers_w = self._word_to_gens_powers(w)
            native = self.from_cone_label(gens_w, powers_w)
            phase_out = self.cone_label_phase(gens_w, powers_w)
            return ("leaf", {native: self._q_one(-phase_out)})
        return ("node", [(co, tuple(nw)) for (co, nw) in result])

    def _collapse_rho2_orbits_in_element(
        self, alg: "KAlgebra", element: Element
    ) -> Element:
        """Replace each label in `element.terms` by the canonical
        representative of its ρ²-orbit (via
        `alg._canonical_rho2_orbit_rep`), summing coefficients.

        `Tr(ρ²(x)) = Tr(x)` is a KAlgebra axiom, so this preserves the
        trace value when `element` is interpreted as a trace-seed
        expression.  By calling this at every return point of
        `simplify_trace_via_cone_data`, we guarantee that downstream
        Layer-2 (`ConeKAlgebra._trace_residual`) only ever receives
        canonical orbit representatives — the subclass cannot
        accidentally violate ρ²-cyclicity by returning different
        residual values on ρ²-related seeds.

        Delegation.  The actual ρ²-orbit canonicalisation lives in
        `KAlgebra._canonical_rho2_orbit_rep` (a general utility
        independent of cone structure).  Default impl is an orbit walk
        with safety bound; algebras with infinite ρ²-orbits (e.g.
        U(1)-gauged theories with torus drift) must override that
        method with a closed-form drift-quotient.

        Short-circuit: if `alg.rho_squared_is_identity()` is True, the
        canonicalisation is a no-op and we return `element` unchanged.
        """
        if alg.rho_squared_is_identity():
            return element
        folded: dict[Label, LaurentPoly] = {}
        canon_cache: dict[Label, Label] = {}
        for seed, coeff in element.terms.items():
            canon = canon_cache.get(seed)
            if canon is None:
                canon = alg._canonical_rho2_orbit_rep(seed)
                canon_cache[seed] = canon
            if canon in folded:
                folded[canon] = folded[canon] + coeff
            else:
                folded[canon] = coeff
        return Element(
            {lbl: c for lbl, c in folded.items() if not c.is_zero()}
        )

    def _tagged_cyclicity_round(
        self,
        alg: "KAlgebra",
        c: LaurentPoly,
        word: list[Label],
    ) -> list[tuple[LaurentPoly, list[Label]]] | str:
        """Run tagged cyclicity on `word` with accumulated coefficient `c`.

        Tries Form B first (pop front, ρ⁻², push back; slide back
        leftward).  If no Plücker fires after `cycle_period_bound`
        cycles, tries Form A (pop back, ρ², push front; slide front
        rightward).  If both forms exhaust, returns the sentinel
        `"exhausted"`.

        Returns a list of `(coeff, new_word)` work items on success
        (typically one item per Plücker daughter).
        """
        if self.bilateral_layer1:
            return self._tagged_cyclicity_round_bilateral(alg, c, word)
        for form in ("B", "A"):
            result = self._tagged_cyclicity_one_form(alg, c, list(word), form)
            if result is not None:
                return result
        return "exhausted"

    def _tagged_cyclicity_round_bilateral(
        self,
        alg: "KAlgebra",
        c: LaurentPoly,
        word: list[Label],
    ) -> list[tuple[LaurentPoly, list[Label]]] | str:
        """Bilateral tagged cyclicity: run Form B (ρ⁻²) and Form A (ρ²)
        and return the Plücker daughters from whichever fires a collision
        in **fewer cycles** ("which Plücker first").  Ties go to Form B,
        so a configuration where both collide on the first cycle behaves
        identically to the default Form-B-first reducer.

        Correctness is direction-independent: both forms are valid
        ρ²-cyclicity identities that reduce `word` to the *same* trace,
        so this only re-routes the search, never the result.  Enabled by
        the `bilateral_layer1` flag; off by default.
        """
        best: tuple[int, list[tuple[LaurentPoly, list[Label]]]] | None = None
        for form in ("B", "A"):
            res = self._tagged_cyclicity_one_form(
                alg, c, list(word), form, track_rounds=True
            )
            if res is None:
                continue
            n_round, daughters = res
            # Strict `<` with B evaluated first ⇒ ties resolve to Form B.
            if best is None or n_round < best[0]:
                best = (n_round, daughters)
        return "exhausted" if best is None else best[1]

    def _tagged_cyclicity_one_form(
        self,
        alg: "KAlgebra",
        c: LaurentPoly,
        word: list[Label],
        form: str,
        track_rounds: bool = False,
    ) -> (
        list[tuple[LaurentPoly, list[Label]]]
        | tuple[int, list[tuple[LaurentPoly, list[Label]]]]
        | None
    ):
        """Run tagged cyclicity in the specified form.  Returns the
        Plücker-split work items on success, `None` if no Plücker
        fires within the cycle bound.  When `track_rounds` is set, the
        success value is instead `(n_cycles, work_items)` where
        `n_cycles` is the 1-based cycle on which the Plücker fired — the
        cost signal the bilateral driver races on.

        Quantum-torus generalisation.  In a quantum-torus cone `ρ∓²` of a
        single conventional mult-gen is **torus-dressed**:
        `ρ⁻²(L_g) = L_{g'·E^d}` for a conventional gen `g'` and a torus
        drift `E^d` (e.g. `U1A1Aodd`'s wrap chords).  The image is then a
        *multi-letter* canonical label `(g', E, …, E)`.  Rather than bail
        (the monomial-only behaviour), we:

          * convert that canonical label to its literal word via
            `cone_label_phase` — `L_{g'·E^d} = q^{phase}·ℓ(g', E,…,E)` —
            and fold `q^{phase}` into the running coefficient (the
            canonical→literal factor; **zero** for a single-gen
            monomial-cone image, so monomial cones are untouched);
          * splice the whole literal block in, and slide the *conventional*
            letter `g'` (located by the torus test — torus letters
            q-commute with everything and never Plücker) toward its
            collision.

        A genuine *multi-conventional-gen* `ρ²` image (more than one
        non-torus letter) is still outside the engine's scope → bail
        (`None`)."""
        max_rounds = self.cycle_period_bound() + 1
        c_local = c
        for _round in range(1, max_rounds + 1):
            # CYCLE step.
            if form == "B":
                # Form B: Tr(L_g · X) = Tr(X · ρ⁻²(L_g)).  Pop front g,
                # replace by ρ⁻²(g), push the literal image to the back.
                tag = word.pop(0)
                tag_native = self.from_cone_label(frozenset({tag}), {tag: 1})
                shifted_native = alg.rho_inverse(alg.rho_inverse(tag_native))
                unit = self._rho2_twist_unit(alg, tag_native, -1)
                if unit is not None:
                    c_local = c_local * unit
                shifted_gens, shifted_powers = self.to_cone_label(shifted_native)
                shifted_word = self._cone_label_to_word(
                    shifted_gens, shifted_powers
                )
                # Canonical→literal phase: L_{shifted_native} =
                # q^{phase}·ℓ(shifted_word).  Zero for a single-gen image.
                c_local = c_local * self._q_one(
                    self.cone_label_phase(shifted_gens, shifted_powers)
                )
                base = len(word)
                word.extend(shifted_word)
                start = self._block_slide_start(word, base, len(word), form)
                if start is None:
                    return None      # multi-conventional ρ² image: bail
                if start == "noslide":
                    # Popped a ρ²-fixed torus letter; it stays at the back.
                    # Cycle again to reach a conventional letter.
                    continue
                # SLIDE step: slide the conventional letter leftward.
                for pos in range(start, 0, -1):
                    partner = word[pos - 1]
                    tagged = word[pos]
                    if not self.q_commute(partner, tagged):
                        # Cross-cone collision: apply cross_product.
                        splice = self._splice_cross_product(
                            alg, c_local, word, pos - 1, pos
                        )
                        return (_round, splice) if track_rounds else splice
                    # q-commute: swap with cocycle phase.
                    # L_partner · L_tagged = q^{2 c(partner, tagged)} · L_tagged · L_partner
                    # Inside a trace position-swap incurs the SAME phase
                    # (trace cyclicity preserves equality of expressions
                    # before/after swap).
                    swap_exp = 2 * self.cocycle(partner, tagged)
                    c_local = c_local * self._q_one(swap_exp)
                    word[pos - 1], word[pos] = word[pos], word[pos - 1]
                # No collision this round; loop and cycle again.
            else:
                # Form A: Tr(X · L_g) = Tr(ρ²(L_g) · X).  Pop back g,
                # replace by ρ²(g), push the literal image to the front.
                tag = word.pop()
                tag_native = self.from_cone_label(frozenset({tag}), {tag: 1})
                shifted_native = alg.rho(alg.rho(tag_native))
                unit = self._rho2_twist_unit(alg, tag_native, +1)
                if unit is not None:
                    c_local = c_local * unit
                shifted_gens, shifted_powers = self.to_cone_label(shifted_native)
                shifted_word = self._cone_label_to_word(
                    shifted_gens, shifted_powers
                )
                c_local = c_local * self._q_one(
                    self.cone_label_phase(shifted_gens, shifted_powers)
                )
                block = list(shifted_word)
                word[:0] = block                       # prepend
                start = self._block_slide_start(word, 0, len(block), form)
                if start is None:
                    return None
                if start == "noslide":
                    # Popped a ρ²-fixed torus letter; it stays at the front.
                    # Cycle again to reach a conventional letter.
                    continue
                # SLIDE step: slide the conventional letter rightward.
                for pos in range(start, len(word) - 1):
                    partner = word[pos + 1]
                    tagged = word[pos]
                    if not self.q_commute(tagged, partner):
                        splice = self._splice_cross_product(
                            alg, c_local, word, pos, pos + 1
                        )
                        return (_round, splice) if track_rounds else splice
                    swap_exp = 2 * self.cocycle(tagged, partner)
                    c_local = c_local * self._q_one(swap_exp)
                    word[pos], word[pos + 1] = word[pos + 1], word[pos]
        return None

    def _block_slide_start(
        self, word: list[Label], lo: int, hi: int, form: str
    ) -> int | None:
        """Index in the freshly-spliced block `word[lo:hi]` (the literal
        ρ∓²-image of a single popped letter) of the conventional (non-
        torus) letter to slide.

        Torus letters (those with a declared `_torus_inverse_letter`)
        q-commute with everything and never Plücker, so the active letter
        is the unique conventional one.  Returns:

          * its index, for the normal single-conventional case (a plain
            MonomialCone block is one conventional letter → returns it,
            identical to the prior single-tag slide);
          * the sentinel `"noslide"` when the block is all-torus (e.g. a
            popped `E`, which is ρ²-fixed): the (ρ∓²-imaged) torus letter
            is left parked at the far end where it was just spliced — NOT
            slid back into the word — so the next cycle round reaches a
            conventional letter instead of churning the ρ²-fixed torus
            generator forever (the failure mode when torus letters sort to
            a word end and block both Form B and Form A);
          * `None` when the block has *more than one* conventional letter
            (a genuine multi-gen ρ² image the engine doesn't handle → the
            caller bails)."""
        conv = [
            i for i in range(lo, hi)
            if self._torus_inverse_letter(word[i]) is None
        ]
        if len(conv) > 1:
            return None
        if conv:
            return conv[0]
        return "noslide"

    def _exhausted_cross_reduce(
        self, alg: "KAlgebra", c: LaurentPoly, word: list[Label]
    ) -> list[tuple[LaurentPoly, list[Label]]]:
        """Resolve a multi-cone literal `word` (one ρ²-cyclicity could not
        fold) by applying `cross_product` at a crossing pair.

        Finds the minimal-gap non-q-commuting pair `(i, j)`; by minimality
        the in-between letters q-commute with `word[i]`, so it slides
        `word[i]` rightward to be adjacent to `word[j]` (accumulating the
        cocycle phase) and splices the Plücker.  Returns the daughter work
        items (re-enqueued and reduced further by the caller).  The Plücker
        strictly lowers the crossing complexity, guaranteeing progress."""
        w = list(word)
        best: tuple[int, int] | None = None
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                if not self.q_commute(w[i], w[j]):
                    if best is None or (j - i) < (best[1] - best[0]):
                        best = (i, j)
                    break          # nearest crossing partner for this i
        if best is None:
            return []              # actually single-cone (defensive no-op)
        i, j = best
        c_local = c
        for p in range(i, j - 1):
            swap_exp = 2 * self.cocycle(w[p], w[p + 1])
            c_local = c_local * self._q_one(swap_exp)
            w[p], w[p + 1] = w[p + 1], w[p]
        return self._splice_cross_product(alg, c_local, w, j - 1, j)

    def _splice_cross_product(
        self,
        alg: "KAlgebra",
        c: LaurentPoly,
        word: list[Label],
        idx_a: int,
        idx_b: int,
    ) -> list[tuple[LaurentPoly, list[Label]]]:
        """Apply `cross_product(word[idx_a], word[idx_b])` and splice each
        daughter back into the word at positions [idx_a, idx_b]."""
        g = word[idx_a]
        h = word[idx_b]
        prefix = word[:idx_a]
        suffix = word[idx_b + 1:]
        out: list[tuple[LaurentPoly, list[Label]]] = []
        for (coeff, replacement) in self.cross_product(g, h):
            spliced = prefix + list(replacement) + suffix
            out.append((c * coeff, spliced))
        return out

    def cycle_period_bound(self) -> int:
        """Upper bound on the number of `ρ²` cycles needed before a
        tagged letter exits any cone in this algebra.

        Default: 64.  Subclasses that know their `ρ` has period `H`
        (e.g. pentagon `H=5`, A1A2k `H=2k+3`) should override to that
        value.  The bound only affects efficiency, not correctness —
        the algorithm tries Form A as a fallback if Form B exhausts.
        """
        return 64

    def trace_vanishes_by_rho2_fixed_factor(
        self,
        alg: "KAlgebra",
        gens: frozenset[Label],
        powers: dict[Label, int],
    ) -> bool:
        """Universal trace-vanishing rule: ``Tr(L_{native}) = 0`` if
        some factor `g` of the cone-monomial is ρ²-fixed at the
        algebra-label level (``ρ²(L_g) = L_g``) AND has non-zero net
        q-commute cocycle with the other factors of the monomial.

        Derivation.  For any ρ²-fixed factor `g` and the rest `O`:

            Tr(g · O)  =  Tr(O · ρ²(g))                  (ρ²-cyclicity)
                       =  Tr(O · g)                       (g is ρ²-fixed)
                       =  q^{-2 · net_c(g)} · Tr(g · O)   (q-commute slide)

        (The general ρ²-twisted-cyclicity axiom twists the other way;
        the first line is valid here precisely because `g` is ρ²-fixed,
        so `ρ^{±2}(g) = g` and the direction of the twist is
        immaterial.)

        where ``net_c(g) = Σ_{h ≠ g} powers[h] · cocycle(g, h)``.
        Hence ``(1 - q^{-2 net_c}) Tr(g · O) = 0``, forcing ``Tr = 0``
        whenever ``net_c ≠ 0``.

        This is a **universal** check (applies to every KAlgebra / cone
        structure).  For algebras whose mult-gens are not ρ²-fixed
        (e.g. Pentagon / Heptagon / A1A2k, where ρ² has finite non-
        trivial order on every L_i), the check is a no-op (always
        returns False).  For QT-style algebras with ρ²-fixed torus
        directions (Sqed1, U1Square, U(1)Hex, ...) it short-circuits
        a large class of trace evaluations.
        """
        for g, exp_g in powers.items():
            if exp_g <= 0:
                continue
            # Is g ρ²-fixed at the algebra-label level?
            g_label = self.from_cone_label(frozenset({g}), {g: 1})
            if alg.rho(alg.rho(g_label)) != g_label:
                continue
            # g is ρ²-fixed.  Compute net cocycle with the rest.
            net_c = 0
            for h, exp_h in powers.items():
                if h == g:
                    continue
                net_c += exp_h * self.cocycle(g, h)
            if net_c != 0:
                return True
        return False

    def tagged_cyclicity_trigger(self, word: tuple[Label, ...]) -> bool:
        """Structural trigger: True iff tagged-cyclicity *might*
        succeed for this word, i.e. some power of ρ² (positive or
        negative) pushes at least one factor of `word` out of the cone
        of the other factors, exposing a Plücker.

        Used by `simplify_trace_via_cone_data` to skip the tagged-
        cyclicity round entirely when no Plücker is achievable —
        saving wasted cycle budget on quantum-torus algebras where ρ²
        preserves the cone-of-others for every factor.

        Default: True (try cyclicity).  Matches the monomial-cone
        behaviour of Pentagon / Heptagon / A1A2k where ρ² has finite
        order on mult-gens and can push them across cones.

        Subclasses with quantum-torus structure (e.g. `U1SquareConeData`)
        should override to False when no ρ² power moves any factor out
        of the cone of the others.  Concretely, for U1SquareConeData all
        factors live in ρ²-fixed cones with torus-direction drift, so
        the trigger is unconditionally False.
        """
        return True

    # -- verifier hooks ---------------------------------------------------

    def verify_roundtrip(self, native_label: Label) -> bool:
        """Check `from_cone_label(*to_cone_label(x)) == x`."""
        gens, powers = self.to_cone_label(native_label)
        # Sanity: gens is the keys of powers, powers values are strictly
        # positive, and gens is pairwise q-commuting.
        if set(powers.keys()) != set(gens):
            return False
        if any(p <= 0 for p in powers.values()):
            return False
        gs = list(gens)
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                if not self.q_commute(gs[i], gs[j]):
                    return False
        return self.from_cone_label(gens, powers) == native_label

    def verify_consistent_with_multiply(
        self, alg: "KAlgebra", a: Label, b: Label
    ) -> bool:
        """Check `derived_multiply(a, b) == alg.multiply(a, b)`."""
        return self.derived_multiply(a, b) == alg.multiply(a, b)

    # -- internal helpers -------------------------------------------------

    def _cone_label_to_word(
        self, gens: frozenset[Label], powers: dict[Label, int]
    ) -> tuple[Label, ...]:
        """(gens, powers) → canonical-ordered word."""
        if not gens:
            return ()
        order = self.canonical_cone_order(gens)
        word: tuple[Label, ...] = ()
        for g in order:
            word = word + (g,) * powers.get(g, 0)
        return word

    def _word_cone(
        self, word: tuple[Label, ...]
    ) -> frozenset[Label] | None:
        """Frozenset of distinct letters iff they're pairwise q-commuting;
        else None.  Empty word → empty frozenset."""
        distinct = list(dict.fromkeys(word))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if not self.q_commute(distinct[i], distinct[j]):
                    return None
        return frozenset(distinct)

    def _sort_within_cone(
        self,
        c: LaurentPoly,
        word: tuple[Label, ...],
        cone: frozenset[Label],
    ) -> tuple[LaurentPoly, tuple[Label, ...]]:
        """Bubble-sort `word` into the cone's canonical order, accumulating
        cocycle q-phase on each adjacent swap.

        Result: `(c · q^{phase}, sorted_word)`.
        """
        if len(word) <= 1:
            return c, word
        order = self.canonical_cone_order(cone)
        rank = {g: i for i, g in enumerate(order)}
        w = list(word)
        phase_exp = 0
        n = len(w)
        for _ in range(n):
            changed = False
            for k in range(n - 1):
                if rank[w[k]] > rank[w[k + 1]]:
                    phase_exp += 2 * self.cocycle(w[k], w[k + 1])
                    w[k], w[k + 1] = w[k + 1], w[k]
                    changed = True
            if not changed:
                break
        return c * self._q_one(phase_exp), tuple(w)

    def _word_to_gens_powers(
        self, word: tuple[Label, ...]
    ) -> tuple[frozenset[Label], dict[Label, int]]:
        """Convert a word to (gens, powers).  Assumes word is single-cone.

        For quantum-torus ConeData with `_torus_inverse_letter`
        overridden, paired torus letters `v_p^a · v_n^b` collapse to
        the net `v_p^{a-b}` (or `v_n^{b-a}` if negative), since
        `v_p · v_n = 1` in the algebra.  This implements the Laurent-
        cone semantics: at the cone-monomial boundary, the bidirectional
        torus letters cancel; downstream `from_cone_label` then sees
        only one of v_p / v_n with positive power.
        """
        powers: dict[Label, int] = {}
        for g in word:
            powers[g] = powers.get(g, 0) + 1
        # Collapse torus inverse pairs (no-op when `_torus_inverse_letter`
        # is the default None → no torus structure).
        for g in list(powers.keys()):
            inv = self._torus_inverse_letter(g)
            if inv is None or inv not in powers or inv == g:
                continue
            # Process each pair only once.
            if g > inv:
                continue
            a, b = powers[g], powers[inv]
            net = a - b
            if net > 0:
                powers[g] = net
                del powers[inv]
            elif net < 0:
                powers[inv] = -net
                del powers[g]
            else:
                del powers[g]
                del powers[inv]
        return frozenset(powers.keys()), powers

    def _torus_inverse_letter(self, g: Label) -> "Label | None":
        """If `g` is a torus mult-gen with an inverse letter in this
        cone-data's mult-gen set, return the inverse letter; else None.

        Default: None for all gens (= no torus structure, the monomial /
        Pentagon-style behaviour).

        Quantum-torus `ConeData` subclasses (e.g. `U1SquareConeData`)
        override to declare the v / v_inv pairing.  When set, the
        `_word_to_gens_powers` step collapses paired letters according
        to `v_p · v_n = 1` (Laurent-cone cancellation).
        """
        return None

    # -- Cone-object API (lifts the cone-as-frozenset convention) ---------

    def iter_cones(self) -> "Iterable[Cone]":
        """Iterate `Cone` objects covering the multiplicative-generator
        set of this `ConeData`.

        Default impl: requires the subclass to expose a `cones()` method
        returning frozensets of mult-gens (as `FiniteConeData` does);
        wraps each frozenset in a plain (monomial) `Cone`.

        Subclasses with non-monomial canonical bases (e.g. character
        cones) must override to yield `Cone`s with the appropriate
        `torus_gens` / `char_gens` partition.
        """
        if not hasattr(self, "cones"):
            raise NotImplementedError(
                f"{type(self).__name__}.iter_cones(): provide a `cones()` "
                f"method returning frozensets of mult-gens (default "
                f"monomial-Cone wrapping), or override iter_cones() directly."
            )
        for gens_fs in self.cones():
            yield Cone(self, frozenset(gens_fs))

    def cone_of_label(self, native_label: Label) -> "Cone":
        """The `Cone` containing the mult-gen support of `native_label`.

        Default impl: locates a cone whose `mult_gens()` contains the
        gens returned by `to_cone_label(native_label)`.  Identity label
        (empty gens) returns the first cone (caller should not rely on
        the cone for identity-label semantics).
        """
        gens_fs, _ = self.to_cone_label(native_label)
        if not gens_fs:
            for c in self.iter_cones():
                return c
            raise ValueError(
                "cone_of_label: no cones available for identity label"
            )
        for c in self.iter_cones():
            if gens_fs.issubset(c.mult_gens()):
                return c
        raise ValueError(
            f"cone_of_label: no cone contains gens {gens_fs} "
            f"for label {native_label!r}"
        )


# ----------------------------------------------------------------------
# Finite-cone convenience subclass
# ----------------------------------------------------------------------


class FiniteConeData(ConeData):
    """`ConeData` specialization for K_𝖖-algebras with a finite, enumerable
    multiplicative-generator set and finitely many cones.

    Subclasses supply `mult_gens()` (the global mult-gen set) and
    `cones()` (a sequence of frozensets — maximal q-commuting subsets,
    or any covering set of compatible subsets); `q_commute` is derived
    by membership lookup.  Subclasses still supply the bijection,
    `cocycle`, `cross_product`, and (when default `sorted` order doesn't
    match the convention) `canonical_cone_order`.
    """

    @abstractmethod
    def mult_gens(self) -> Sequence[Label]:
        """The global, finite set of multiplicative generators."""

    @abstractmethod
    def cones(self) -> Sequence[frozenset[Label]]:
        """Maximal q-commuting subsets of `mult_gens()`.  Used to derive
        the default `q_commute` predicate."""

    def q_commute(self, g: Label, h: Label) -> bool:
        if g == h:
            return True
        return any(g in C and h in C for C in self.cones())


# ----------------------------------------------------------------------
# Cone class hierarchy
# ----------------------------------------------------------------------
#
# A `Cone` is a maximal q-commuting-up-to-cocycle subalgebra of a
# `KAlgebra`.  Each cone has multiplicative generators (PBW basis is
# q-normal-ordered monomials in those gens) and a canonical basis of
# A_q[T] that lives in the cone (the labels for which `to_cone_label`
# returns this cone's mult-gens).  Inside one cone these two bases
# differ by a change of variables:
#
#   * For "monomial cones" the change of variables is the identity:
#     each canonical-basis element IS a q-normal-ordered PBW monomial.
#     Current cases: Pentagon, Heptagon, U1Hexagon, A1A2k.
#
#   * For "quantum-torus cones" (a `Cone` with `torus_gens` non-empty)
#     the change of variables is still the identity, but a marked subset
#     of mult-gens is invertible (`v · v_inv = 1`), so the cone carries
#     a quantum-torus factor and ρ can shear the torus directions.
#     In use: U1A1Aodd(k).
#
#   * For "character cones" (a `Cone` with `char_gens` non-empty) the
#     canonical basis coincides with the irreducible characters χ_k of a
#     q-deformed rep ring; the PBW basis is monomials in the fundamental
#     gen χ_1.  The two bases are related by a Chebyshev / Weyl-character-
#     formula change of variables.  Use case: SU(2)-flavoured A1D3..A1D7,
#     U1A1D2/4, pure-SU(2) Wilson-line cones, SU(N) Schur-poly cones.
#     Implemented for SU(2): the `char_gens` partition (Chebyshev
#     `su2_*_to_*`); in use for the pure-SU(2) gauge-Wilson direction.
#     (SU(N) Schur wires a different change-of-basis pair.)

ConePolynomial = dict[tuple[Label, ...], LaurentPoly]
"""A Z[q^±]-linear combination of canonical-ordered PBW words (words
in mult-gens) within a single cone.  Keys are words (tuples of mult-gens
in cone-canonical order); values are LaurentPoly coefficients.

For monomial cones, every canonical-basis label maps to a single-monomial
ConePolynomial of the form `{word: LaurentPoly({0: 1})}`.  For character
cones, the canonical-basis label `χ_k` maps to the Chebyshev expansion
`U_k(χ_1)` — a true polynomial in the mult-gen word for `χ_1`.
"""


class Cone:
    """A maximal q-commuting-up-to-cocycle subalgebra of a `KAlgebra`.

    Each cone has:
      * `mult_gens()` — the multiplicative generators of the cone
        (PBW basis is q-normal-ordered monomials in these);
      * a parent `ConeData` (for q-commute / cocycle / ordering data);
      * a `pbw_to_canonical` map (PBW monomial → canonical-basis
        Element) and its inverse `canonical_to_pbw`.

    `Cone` is **one concrete class parameterized by a partition of its
    mult-gens** into orthogonal, composable kinds:

      * **torus** gens (`torus_gens`) — invertible directions
        (`v · v_inv = 1`); the cone subalgebra carries a quantum-torus
        factor.
      * **character** gens (`char_gens`) — each a fundamental SU(2)
        character `χ_1`; the change of basis on these is the Chebyshev /
        Weyl relation `χ_k = U_k(χ_1)`.
      * **monomial** gens (the remainder, `monomial_gens()` =
        `mult_gens − torus_gens − char_gens`) — the change of basis is
        the identity.

    The kinds compose orthogonally, so mixed cones (e.g. torus +
    character together) are constructible directly via
    `Cone(parent, mult_gens, torus_gens=…, char_gens=…)`.  The partition
    is carried as data and queried with `is_monomial()` /
    `is_quantum_torus()` / `is_character()`; there are no per-kind
    subclasses (the former `MonomialCone` / `QTCone` / `CharacterCone`
    were thin partition-fixers, folded into this class).

    The PBW ↔ canonical change of variables dispatches on the
    partition: an empty `char_gens` gives the identity (monomial) change
    of basis; a nonempty `char_gens` gives the Chebyshev expansion on the
    marked char-gens, with the remaining gens carried through monomially.

    ρ semantics by partition
    ------------------------
    * **Monomial cone** (no torus, no char gens): the mult-gens are a
      **ρ-invariant subset** of the canonical basis — ρ permutes them as a
      set, mapping each mult-gen to another single mult-gen (possibly in a
      different cone).  Matches Pentagon / Heptagon / A1A2k, where ρ shifts
      the index `i ↦ i+2` (mod H); the set-theoretic ρ-orbit on cones is
      finite (period dividing the global ρ-period).

    * **Quantum-torus cone** (`torus_gens()` non-empty): the mult-gens are
      NOT necessarily ρ-invariant — ρ can map a conventional gen `g` to a
      *PBW composite* `g' · v^k` (another conventional `g'` times a torus
      power), not a single mult-gen.  ρ permutes QT cones in a finite
      cycle; the **cone-fixing power** is ρ^H or ρ^{H/2} (per-algebra; e.g.
      ρ⁴ for U1Square with H=4, ρ⁶ for U1Hexagon), acting inside each cone
      as a **torus-direction shear** parameterised by the conventional
      gen's magnetic charge m — e.g. Sqed1 `ρ²((m,n)) = (m, n+m)`,
      `ρ⁴((m,n)) = (m, n+2m)`.  The Layer-1 ρ²-fold walks ρ²-orbits (each
      visiting H/2 cones before returning with accumulated v-drift) and
      canonicalises by quotienting torus exponents by that drift.  The
      torus cancellation `v · v_inv = 1` is encoded by
      `cross_product(v, v_inv) = [(LaurentPoly.one(), ())]` at the parent
      `ConeData`; only one of `v`/`v_inv` appears (positive power) in a
      canonical label.  The universal `cone_label_phase`
      `phase = -Σ_{i<j} c(a_i, a_j)` (pinned by bar-invariance) works
      identically for QT cones on the split-letter word, so verify-roundtrip
      and orthonormality verifiers need no modification.
    """

    def __init__(
        self,
        parent: "ConeData",
        mult_gens: frozenset[Label],
        torus_gens: frozenset[Label] = frozenset(),
        char_gens: frozenset[Label] = frozenset(),
    ):
        self._parent = parent
        self._mult_gens = frozenset(mult_gens)
        self._torus_gens = frozenset(torus_gens)
        self._char_gens = frozenset(char_gens)
        if not self._torus_gens <= self._mult_gens:
            raise ValueError(
                f"Cone: torus_gens {self._torus_gens - self._mult_gens} are "
                f"not in mult_gens {self._mult_gens}"
            )
        if not self._char_gens <= self._mult_gens:
            raise ValueError(
                f"Cone: char_gens {self._char_gens - self._mult_gens} must be "
                f"a subset of mult_gens {self._mult_gens}"
            )
        if self._torus_gens & self._char_gens:
            raise ValueError(
                f"Cone: torus_gens and char_gens must be disjoint; both "
                f"contain {self._torus_gens & self._char_gens}"
            )

    def mult_gens(self) -> frozenset[Label]:
        """The multiplicative generators of this cone."""
        return self._mult_gens

    def parent_data(self) -> "ConeData":
        """The `ConeData` this cone belongs to (cocycle / ordering lookup)."""
        return self._parent

    def torus_gens(self) -> frozenset[Label]:
        """The subset of `mult_gens()` tagged as quantum-torus
        directions.  Cancellation `v · v_inv = 1` for v in this set is
        encoded via the parent's `cross_product`."""
        return self._torus_gens

    def char_gens(self) -> frozenset[Label]:
        """The subset of `mult_gens()` tagged as SU(2) character (`χ_1`)
        directions; the canonical basis on these is `{χ_k}`."""
        return self._char_gens

    def conventional_gens(self) -> frozenset[Label]:
        """`mult_gens() - torus_gens()`.  Powers in canonical labels are
        strictly positive for these.  (Quantum-torus compatibility.)"""
        return self._mult_gens - self._torus_gens

    def monomial_gens(self) -> frozenset[Label]:
        """`mult_gens() - torus_gens() - char_gens()`: the remainder gens
        whose canonical ↔ PBW change of basis is the identity."""
        return self._mult_gens - self._torus_gens - self._char_gens

    def is_quantum_torus(self) -> bool:
        """True iff this cone carries a quantum-torus factor (`torus_gens()`
        non-empty).  The successor of the former `isinstance(·, QTCone)`."""
        return bool(self._torus_gens)

    def is_character(self) -> bool:
        """True iff this cone has SU(2) character directions (`char_gens()`
        non-empty).  The successor of the former `isinstance(·, CharacterCone)`."""
        return bool(self._char_gens)

    def is_monomial(self) -> bool:
        """True iff the canonical ↔ PBW change of basis is the identity on
        every gen (no torus, no character directions).  The successor of the
        former `isinstance(·, MonomialCone)`."""
        return not self._torus_gens and not self._char_gens

    def q_commute(self, g: Label, h: Label) -> bool:
        """Delegate to the parent ConeData."""
        return self.parent_data().q_commute(g, h)

    def cocycle(self, g: Label, h: Label) -> int:
        """Delegate to the parent ConeData."""
        return self.parent_data().cocycle(g, h)

    def canonical_cone_order(self) -> tuple[Label, ...]:
        """Linear order on `mult_gens()` for PBW-word formation."""
        return self.parent_data().canonical_cone_order(self._mult_gens)

    def _expand(self, powers: dict[Label, int], table) -> list:
        """Cartesian-product the per-char-gen `table(power)` expansions with the
        monomial (non-char) gens carried through.  Returns `[(full_powers, coeff)]`."""
        mono = {g: p for g, p in powers.items()
                if g not in self._char_gens and p}
        combos = [(dict(mono), 1)]
        for g in self._char_gens:
            p = powers.get(g, 0)
            if p <= 0:
                continue
            exp = table(p)
            combos = [({**d, g: idx}, coeff * c)
                      for (d, coeff) in combos for idx, c in exp.items()]
        return combos

    def pbw_to_canonical(
        self, powers: dict[Label, int]
    ) -> "Element":
        """Map a q-normal-ordered PBW monomial in `mult_gens()` (given
        by its `{gen: power}` dict, powers ≥ 1) to a canonical-basis
        `Element` of the algebra.

        With an empty `char_gens` the PBW monomial IS the canonical basis
        element (up to a phase already handled at the caller level), so
        the return is a single-term `Element`.

        With a nonempty `char_gens` the PBW monomial `χ_1^k` on each
        char-gen corresponds to a Chebyshev/Weyl-character expansion of
        canonical-basis χ_j's.
        """
        if not self._char_gens:
            powers = {g: p for g, p in powers.items() if p > 0}
            if not powers:
                native = self._parent.from_cone_label(frozenset(), {})
            else:
                gens = frozenset(powers.keys())
                native = self._parent.from_cone_label(gens, powers)
            return Element({native: self._parent._q_one(0)})
        powers = {g: p for g, p in powers.items() if p > 0}
        if not powers:
            native = self._parent.from_cone_label(frozenset(), {})
            return Element({native: self._parent._q_one(0)})
        out: dict[Label, LaurentPoly] = {}
        for full, coeff in self._expand(powers, su2_monomial_to_chars):
            full = {g: p for g, p in full.items() if p}
            native = self._parent.from_cone_label(frozenset(full), full)
            add = LaurentPoly({0: coeff})
            out[native] = (out[native] + add) if native in out else add
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def canonical_to_pbw(
        self, native_label: Label
    ) -> ConePolynomial:
        """Inverse of `pbw_to_canonical`: a native canonical-basis label
        → its expansion as a `Z[q^±]`-polynomial in PBW monomials.

        With an empty `char_gens`: returns a single-monomial dict
        `{word: LaurentPoly.one()}` matching `to_cone_label`'s output.

        With a nonempty `char_gens`: returns the multi-term Chebyshev
        expansion.
        """
        if not self._char_gens:
            gens, powers = self._parent.to_cone_label(native_label)
            if not gens:
                return {(): self._parent._q_one(0)}
            order = self._parent.canonical_cone_order(gens)
            word: tuple[Label, ...] = ()
            for g in order:
                word = word + (g,) * powers.get(g, 0)
            return {word: self._parent._q_one(0)}
        gens, powers = self._parent.to_cone_label(native_label)
        if not gens:
            return {(): self._parent._q_one(0)}
        order = self._parent.canonical_cone_order(self._mult_gens)
        out: ConePolynomial = {}
        for full, coeff in self._expand(dict(powers), su2_char_to_monomials):
            word: tuple[Label, ...] = ()
            for g in order:
                word = word + (g,) * full.get(g, 0)
            add = LaurentPoly({0: coeff})
            out[word] = (out[word] + add) if word in out else add
        return {w: c for w, c in out.items() if not c.is_zero()}


def su2_monomial_to_chars(n: int) -> dict[int, int]:
    """The SU(2) fundamental-power → irreducible-character expansion:

        χ_1^n = Σ_{i=0}^{⌊n/2⌋} (C(n,i) − C(n,i−1)) · χ_{n−2i}

    (`χ_1 = μ+μ⁻¹` the spin-½ character, `χ_k = Σ_{i=0}^{k} μ^{k−2i}` the spin-k/2
    character; this is the Clebsch–Gordan `χ_1·χ_k = χ_{k+1}+χ_{k−1}` iterated).
    Returns `{j: coeff}` (the canonical-character content of the monomial χ_1^n).
    E.g. `χ_1^2 = χ_0 + χ_2`, `χ_1^3 = χ_3 + 2χ_1`."""
    from math import comb
    out: dict[int, int] = {}
    for i in range(n // 2 + 1):
        c = comb(n, i) - (comb(n, i - 1) if i >= 1 else 0)
        if c:
            out[n - 2 * i] = c
    return out


def su2_char_to_monomials(k: int) -> dict[int, int]:
    """The inverse: the SU(2) irreducible character χ_k as a polynomial in the
    fundamental χ_1 (the Chebyshev `U_k`), via `χ_k = χ_1·χ_{k−1} − χ_{k−2}`
    (`χ_0 = 1`, `χ_1 = χ_1`).  Returns `{l: coeff}` for `χ_k = Σ_l coeff·χ_1^l`.
    E.g. `χ_2 = χ_1^2 − 1`, `χ_3 = χ_1^3 − 2χ_1`."""
    if k == 0:
        return {0: 1}
    if k == 1:
        return {1: 1}
    a, b = {0: 1}, {1: 1}              # χ_0, χ_1
    for _ in range(2, k + 1):
        new = {l + 1: c for l, c in b.items()}     # χ_1 · χ_{k-1}
        for l, c in a.items():
            new[l] = new.get(l, 0) - c             # − χ_{k-2}
        a, b = b, {l: c for l, c in new.items() if c}
    return b
