"""
kalgebra_iso.py
===============

`KAlgebraIso`: a bijective, structure-preserving correspondence
between two concrete `KAlgebra` instances.  Standalone — does not
modify the `KAlgebra` ABC.

Motivation.  K-algebras with canonical bases admit relatively few
canonical-basis-respecting homomorphisms; in practice the relevant
maps are *isomorphisms* identifying two distinct presentations of
the same algebra (e.g. the chord-polygon presentation
`A1A2kKAlg(k)` and the BPS-quiver presentation `BPSKAlgebra(A_{2k}-quiver)`,
which are isomorphic by a non-trivial chord ↔ F-element identification).
`KAlgebraIso` is the place to record such an iso and to verify that
it preserves every piece of the K-algebra structure.

Construction
------------
A `KAlgebraIso` is specified by *two* maps, mutually inverse, each
sending one algebra's canonical labels to the other's `Element`s:

    forward_label_map : source canonical label → target `Element`
    inverse_label_map : target canonical label → source `Element`

Both are obligatory; an iso has no "forward only" mode.  Each map
is extended `𝖖`-linearly over Elements by `.map` / `.inverse`.

Structure preservation
----------------------
A genuine `KAlgebra` isomorphism preserves every primitive of the
K-algebra contract:

  * unit                 map(1_source) = 1_target
  * multiplication       map(a · b)    = map(a) · map(b)
  * ρ-action             map(ρ_S(a))   = ρ_T(map(a))
  * (and so, automatically) bar involution and ρ²-twisted trace.

`KAlgebraIso` provides `verify_*` methods checking each of these on
a finite sample, in *both* directions (forward and inverse).
`verify_all(samples, pairs)` runs the full battery.

The verifications are necessary but not sufficient on any finite
sample; the user is responsible for picking samples adequate for
their use case (typically: generators, generator-times-generator,
length-2 canonical labels, and a handful of length-3 cases).

Caveat
------
`KAlgebraIso` does *not* derive the label maps from a generator
dictionary on its own — assembling
    forward_label_map(λ_source) = (target image of L_{a_1}) · ··· · (image of L_{a_n})
requires the source algebra's canonical-label-to-generator
decomposition (which differs per concrete `KAlgebra`).  Concrete iso
constructors (e.g. `A1A2kKAlg(k) ↔ BPSKAlgebra(A_{2k})`) are
responsible for assembling the two maps from the underlying
chord ↔ F-element correspondence.
"""
from __future__ import annotations

from typing import Callable, Iterable

from kalgebra import KAlgebra, Element


class KAlgebraIso:
    """A bijective, structure-preserving correspondence between two
    `KAlgebra` instances.

    Parameters
    ----------
    source, target : KAlgebra
        The two K-algebras (presented in possibly different ways).
    forward_label_map : Callable[[tuple], Element]
        Sends each canonical source label to its image as a target
        `Element`.
    inverse_label_map : Callable[[tuple], Element]
        The inverse map (target canonical label → source `Element`).
        Required: an iso is by definition bijective.
    name : str | None
        Optional descriptive name (e.g. ``"A1A2k(k=2) ≅ BPS(A_4)"``).
    """

    def __init__(
        self,
        source: KAlgebra,
        target: KAlgebra,
        forward_label_map: Callable[[tuple], Element],
        inverse_label_map: Callable[[tuple], Element],
        name: str | None = None,
    ):
        if forward_label_map is None or inverse_label_map is None:
            raise ValueError(
                "KAlgebraIso requires both forward_label_map and "
                "inverse_label_map (it is, by definition, a bijection)."
            )
        self.source = source
        self.target = target
        self._forward = forward_label_map
        self._inverse = inverse_label_map
        self.name = name

    # ---- Mapping ------------------------------------------------------

    def map(self, elem: Element) -> Element:
        """Forward image of a source `Element` in the target algebra."""
        return self._extend(elem, self._forward)

    def inverse(self, elem: Element) -> Element:
        """Inverse image of a target `Element` in the source algebra."""
        return self._extend(elem, self._inverse)

    @staticmethod
    def _extend(elem: Element, label_map) -> Element:
        """`𝖖`-linear extension of `label_map` to Elements."""
        out_terms: dict = {}
        for label, coef in elem.terms.items():
            img = label_map(label)
            for sub_lbl, sub_coef in img.terms.items():
                contrib = coef * sub_coef
                if sub_lbl in out_terms:
                    out_terms[sub_lbl] = out_terms[sub_lbl] + contrib
                else:
                    out_terms[sub_lbl] = contrib
        out_terms = {k: v for k, v in out_terms.items() if not v.is_zero()}
        return Element(out_terms)

    # ---- Verifications -----------------------------------------------
    #
    # Each verification checks the property in BOTH directions: the
    # forward map (source → target) and the inverse map
    # (target → source).  Iso-hood demands both.

    def verify_unit(self) -> bool:
        """`map(1_source) == 1_target`  AND  `inverse(1_target) == 1_source`."""
        from laurent_poly import LaurentPoly
        one = LaurentPoly.one()
        src_unit = Element({self.source.identity(): one})
        tgt_unit = Element({self.target.identity(): one})
        return (self.map(src_unit) == tgt_unit
                and self.inverse(tgt_unit) == src_unit)

    def verify_round_trip(
        self,
        source_samples: Iterable[Element],
        target_samples: Iterable[Element],
    ) -> bool:
        """For each `a ∈ source_samples`:  `inverse(map(a)) == a`.
        For each `b ∈ target_samples`:    `map(inverse(b)) == b`."""
        for a in source_samples:
            if self.inverse(self.map(a)) != a:
                return False
        for b in target_samples:
            if self.map(self.inverse(b)) != b:
                return False
        return True

    def verify_multiplicative(
        self,
        source_pairs: Iterable[tuple[Element, Element]],
        target_pairs: Iterable[tuple[Element, Element]],
    ) -> bool:
        """Forward: `map(a·b) == map(a)·map(b)` for each (a,b) in source_pairs.
        Inverse: `inverse(c·d) == inverse(c)·inverse(d)` for each (c,d)
        in target_pairs."""
        for a, b in source_pairs:
            lhs = self.map(_mul(self.source, a, b))
            rhs = _mul(self.target, self.map(a), self.map(b))
            if lhs != rhs:
                return False
        for c, d in target_pairs:
            lhs = self.inverse(_mul(self.target, c, d))
            rhs = _mul(self.source, self.inverse(c), self.inverse(d))
            if lhs != rhs:
                return False
        return True

    def verify_rho_equivariant(
        self,
        source_samples: Iterable[Element],
        target_samples: Iterable[Element],
    ) -> bool:
        """Forward: `map(ρ_S(a)) == ρ_T(map(a))`.
        Inverse: `inverse(ρ_T(b)) == ρ_S(inverse(b))`."""
        for a in source_samples:
            lhs = self.map(_rho(self.source, a))
            rhs = _rho(self.target, self.map(a))
            if lhs != rhs:
                return False
        for b in target_samples:
            lhs = self.inverse(_rho(self.target, b))
            rhs = _rho(self.source, self.inverse(b))
            if lhs != rhs:
                return False
        return True

    def verify_trace_equivariant(
        self,
        source_samples: Iterable[Element],
        target_samples: Iterable[Element],
        K: int = 12,
    ) -> bool:
        """Forward: `Tr_source(a) == Tr_target(map(a))` truncated at `q^K`.
        Inverse: `Tr_target(b) == Tr_source(inverse(b))` truncated at `q^K`."""
        for a in source_samples:
            lhs = _trace_dict(self.source, a, K)
            rhs = _trace_dict(self.target, self.map(a), K)
            if lhs != rhs:
                return False
        for b in target_samples:
            lhs = _trace_dict(self.target, b, K)
            rhs = _trace_dict(self.source, self.inverse(b), K)
            if lhs != rhs:
                return False
        return True

    def verify_maps_section_to_section_1drep(
        self,
        source_sections: Iterable,
        target_sections: Iterable,
    ) -> bool:
        """A flavoured iso must send each **section** label (a `forget()`-basis
        label = unflavoured canonical) to a single canonical dressed by a
        **1-dimensional rep**: `forward(L_s) == χ · L_{s'}` with `dim(χ) == 1`
        (the Λ-torsor freedom), and symmetrically for `inverse`.

        Checked through the lift coordinate: `forward(L_s)` must be a single
        canonical `L_b` with `target.r_label_decompose(b) = (s', w)` and
        `target.coefficient_ring().dim(w) == 1`.  Additive beside the strict
        full-label battery; requires both algebras to implement
        `r_label_decompose`."""
        return (
            self._sections_to_1drep(self.target, self._forward, source_sections)
            and self._sections_to_1drep(
                self.source, self._inverse, target_sections)
        )

    @staticmethod
    def _sections_to_1drep(tgt, label_map, sections) -> bool:
        for s in sections:
            img = label_map(s)                  # section label -> target Element
            terms = list(img.terms.items())
            if len(terms) != 1:
                return False
            b, _coef = terms[0]
            _s2, w = tgt.r_label_decompose(b)
            if tgt.coefficient_ring().dim(w) != 1:
                return False
        return True

    def verify_all(
        self,
        source_samples: Iterable[Element],
        target_samples: Iterable[Element],
        source_pairs: Iterable[tuple[Element, Element]],
        target_pairs: Iterable[tuple[Element, Element]],
        trace_K: int = 12,
    ) -> dict:
        """Run unit / round-trip / multiplicative / ρ-equivariant /
        trace-equivariant verifications and return a `{check_name: bool}`
        summary.  `trace_K` is the truncation order for trace comparison."""
        source_samples = list(source_samples)
        target_samples = list(target_samples)
        return {
            "unit": self.verify_unit(),
            "round_trip":
                self.verify_round_trip(source_samples, target_samples),
            "multiplicative":
                self.verify_multiplicative(source_pairs, target_pairs),
            "rho_equivariant":
                self.verify_rho_equivariant(source_samples, target_samples),
            "trace_equivariant":
                self.verify_trace_equivariant(
                    source_samples, target_samples, K=trace_K),
        }

    def __repr__(self) -> str:
        nm = (self.name
              or f"{type(self.source).__name__} ↔ {type(self.target).__name__}")
        return f"KAlgebraIso[{nm}]"

    # ---- Algebraic operations on isos --------------------------------
    # `KAlgebraIso` instances form a groupoid: identities, inverses, and
    # composition.  These primitives let isos be combined directly,
    # rather than re-hand-rolling label maps each time.

    @classmethod
    def identity(cls, alg: KAlgebra, name: str | None = None) -> "KAlgebraIso":
        """The identity iso `alg ↔ alg`: each canonical label maps to
        itself as a single-term `Element` with coefficient `1 ∈ R[q^±]`."""
        return cls.identity_on_labels(alg, alg,
                                      name=name or f"id_{type(alg).__name__}")

    @classmethod
    def identity_on_labels(
        cls, source: KAlgebra, target: KAlgebra, name: str | None = None,
    ) -> "KAlgebraIso":
        """The identity-on-labels iso between two *presentations sharing a
        canonical label set* (e.g. a `BPSKAlgebra` and a directional
        RG presentation of the same theory, whose UV labels are the same
        `Γ`-tuples by the apex labelling — decisions A9).  Each label maps
        to itself with coefficient `1`; the mathematical content lives
        entirely in the `verify_*` battery run against it."""
        from laurent_poly import LaurentPoly
        one = LaurentPoly.one()

        def _id(label):
            return Element({label: one})

        return cls(
            source, target,
            forward_label_map=_id,
            inverse_label_map=_id,
            name=name or (f"id-labels[{type(source).__name__} ↔ "
                          f"{type(target).__name__}]"),
        )

    def invert(self) -> "KAlgebraIso":
        """Return the inverse iso `target ↔ source`.  Forward and
        inverse label maps are swapped; no new computation required."""
        inv_name = None
        if self.name is not None:
            inv_name = (
                self.name[1:] if self.name.startswith("inv:")
                else f"inv:{self.name}"
            )
        return KAlgebraIso(
            source=self.target,
            target=self.source,
            forward_label_map=self._inverse,
            inverse_label_map=self._forward,
            name=inv_name,
        )

    def compose(self, other: "KAlgebraIso") -> "KAlgebraIso":
        """Return the composition `other ∘ self`: source = self.source,
        target = other.target, applying `self` first then `other`.

        Requires `self.target is other.source` (the algebras must match
        as Python objects, not just be of the same class — composition
        is on concrete instances).

        Available as `other @ self` via `__matmul__`."""
        if self.target is not other.source:
            raise ValueError(
                f"KAlgebraIso.compose: target of {self!r} (= "
                f"{type(self.target).__name__}) must be the same instance "
                f"as source of {other!r} (= {type(other.source).__name__})"
            )
        fwd_self = self._forward
        fwd_other = other._forward
        inv_self = self._inverse
        inv_other = other._inverse
        extend = self._extend

        def composed_forward(src_label):
            return extend(fwd_self(src_label), fwd_other)

        def composed_inverse(tgt_label):
            return extend(inv_other(tgt_label), inv_self)

        nm = None
        if self.name and other.name:
            nm = f"{other.name} ∘ {self.name}"
        return KAlgebraIso(
            source=self.source,
            target=other.target,
            forward_label_map=composed_forward,
            inverse_label_map=composed_inverse,
            name=nm,
        )

    def __matmul__(self, other: "KAlgebraIso") -> "KAlgebraIso":
        """`other @ self` ≡ `self.compose(other)` — applies `self` first,
        then `other`."""
        return other.compose(self)

    # ---- Cone-data-driven factory -------------------------------------

    @classmethod
    def from_cone_mult_gen_map(
        cls,
        source: KAlgebra,
        target: KAlgebra,
        mult_gen_forward: dict,
        target_label_to_source: Callable,
        name: str | None = None,
    ) -> "KAlgebraIso":
        """Derive a `KAlgebraIso` from a mult-gen correspondence.

        Hypothesis: `source` is a `ConeKAlgebra` (or has a non-None
        `cone_data()`).  Then any source canonical-basis label
        `L_{(gens, powers)}` is the ordered product
        `q^{phase} · ∏ L_{g}^{powers[g]}` in canonical-cone order
        (`source.cone_data().canonical_cone_order`).  Given an image
        `mult_gen_forward[g] ∈ target` for each source mult-gen,
        the forward map is the cone-data-driven lift:

            forward(label) := q^{phase_source(label)}
                              · ∏ mult_gen_forward[g]^{powers[g]}

        computed via `target.multiply_elements`.  The cocycle phase is
        absorbed by the target's own multiply.

        The inverse direction does NOT have a general derivation from
        the mult-gen map alone — it requires inverting a cluster-/
        cone-decomposition (which cone of source mult-gens contains
        the target image as a non-negative integer combination?).
        The caller supplies that as `target_label_to_source`.

        Parameters
        ----------
        source : KAlgebra
            Must have `source.cone_data()` returning a non-None
            `ConeData`.
        target : KAlgebra
            Arbitrary `KAlgebra` (no cone-data requirement).
        mult_gen_forward : dict[Label, Element]
            For each source mult-gen `g`, the `target.Element`
            image — typically `Element({target_label: LaurentPoly.one()})`.
        target_label_to_source : Callable[[Label], Element]
            Inverse direction: given a target canonical-basis label,
            return the source `Element`.  Hand-supplied (cone-data
            inversion requires extra structure, e.g. a cluster
            decomposition).
        """
        cd = source.cone_data()
        if cd is None:
            raise ValueError(
                "from_cone_mult_gen_map: source must have non-None "
                "cone_data()."
            )
        # Validate that mult_gen_forward covers every mult-gen.
        missing = [g for g in cd.mult_gens()
                   if g not in mult_gen_forward]
        if missing:
            raise ValueError(
                f"from_cone_mult_gen_map: mult_gen_forward missing "
                f"images for source mult-gens {missing}."
            )

        from laurent_poly import LaurentPoly
        one = LaurentPoly.one()
        ident_elem = Element({target.identity(): one})

        def forward(source_label):
            gens, powers = cd.to_cone_label(source_label)
            if not gens:
                return ident_elem
            order = cd.canonical_cone_order(gens)
            # Source canonical-basis convention:
            # L_{(gens, powers)} = q^{phase_source} · ∏ L_g^{powers[g]}
            # in canonical-cone-order, where phase = cone_label_phase.
            phase = cd.cone_label_phase(gens, powers)
            result = ident_elem
            for g in order:
                p = powers.get(g, 0)
                if p == 0:
                    continue
                img = mult_gen_forward[g]
                for _ in range(p):
                    result = _mul(target, result, img)
            # Apply the source-side phase to recover the canonical basis
            # element (target multiply already absorbed its own phase).
            if phase != 0:
                result = result * LaurentPoly({phase: 1})
            return result

        return cls(
            source, target,
            forward_label_map=forward,
            inverse_label_map=target_label_to_source,
            name=name,
        )

    @classmethod
    def from_section_map(
        cls,
        source: KAlgebra,
        target: KAlgebra,
        section_map: Callable,
        inverse_section_map: Callable,
        name: str | None = None,
    ) -> "KAlgebraIso":
        """Build a flavoured iso from its action on **sections** alone —
        instead of hand-threading flavour through full labels ("the clumsy
        flavour route").

        `section_map` sends a source section `s` (a `forget()`-basis label) to
        a pair `(s', f)`: the target section `s'` and a 1d-rep `f ∈ Λ` — a
        `coefficient_ring().one_dim_reps()` coordinate (`len(f) ==
        one_dim_rep_rank()`).  The full forward label map is derived from the
        lift-coordinate pair `r_label_decompose` / `r_label_compose`:

            forward(a):
                (s, w)  = source.r_label_decompose(a)     # section + R-irrep
                (s', f) = section_map(s)                   # target section + 1drep
                w'      = embed_one_dim_rep(f) ⊗ w  in R   # Λ-torsor twist
                return  L_{ target.r_label_compose(s', w') }

        `inverse_section_map` gives the mirror.  **Same flavour ring `R` on
        both sides** (an iso is two presentations of one theory; a cross-
        flavour map is `lower_flavour`, not an iso), so the source R-irrep `w`
        carries over and only the 1d-rep `f` twists it.  Requires both
        algebras to implement `r_label_decompose` / `r_label_compose`.
        Degenerates correctly for semisimple / unflavoured `R` (`f = ()`, no
        twist — a plain section bijection)."""
        from laurent_poly import LaurentPoly
        one = LaurentPoly.one()

        def _derive(src, tgt, smap):
            Rt = tgt.coefficient_ring()

            def label_map(a):
                s, w = src.r_label_decompose(a)
                s_t, f = smap(s)
                chi = Rt.embed_one_dim_rep(tuple(f))
                tw = Rt.multiply_basis(chi, w)       # 1drep ⊗ irrep: single irrep
                (w_prime, _mult), = tw.items()
                return Element({tgt.r_label_compose(s_t, w_prime): one})

            return label_map

        return cls(
            source, target,
            forward_label_map=_derive(source, target, section_map),
            inverse_label_map=_derive(target, source, inverse_section_map),
            name=name or (f"section-map[{type(source).__name__} ↔ "
                          f"{type(target).__name__}]"),
        )


# ---- Sample-set helpers --------------------------------------------------


def canonical_iso_samples(
    alg: KAlgebra,
    mult_gens: Iterable | None = None,
    include_pairs: bool = False,
) -> dict:
    """Return a default sample set for `KAlgebraIso.verify_all` driven
    by an algebra's natural generators.

    Returns
    -------
    dict with keys `samples` (list of single-mult-gen `Element`s plus
    the identity) and, if `include_pairs=True`, `pairs` (list of
    (gen × gen) `Element` pairs covering every ordered pair of
    distinct mult-gens, plus (gen × gen) for each gen).

    `mult_gens` defaults to `alg.cone_data().mult_gens()` if the
    algebra has a `cone_data()` method returning non-None; otherwise
    the caller must supply them explicitly.
    """
    from laurent_poly import LaurentPoly
    one = LaurentPoly.one()
    if mult_gens is None:
        cd = getattr(alg, "cone_data", lambda: None)()
        if cd is None:
            raise ValueError(
                "canonical_iso_samples: pass `mult_gens` explicitly when "
                "the algebra has no cone_data."
            )
        mult_gens = list(cd.mult_gens())

    samples = [Element({alg.identity(): one})]
    gen_elements = []
    for g in mult_gens:
        # Map the cone-data mult-gen `g` to its native K-algebra label
        # via `from_cone_label({g}, {g: 1})`, then wrap as Element.
        cd = alg.cone_data()
        native = cd.from_cone_label(frozenset({g}), {g: 1})
        e = Element({native: one})
        gen_elements.append(e)
        samples.append(e)

    out = {"samples": samples}
    if include_pairs:
        pairs = []
        for a in gen_elements:
            for b in gen_elements:
                pairs.append((a, b))
        out["pairs"] = pairs
    return out


def _mul(algebra: KAlgebra, a: Element, b: Element) -> Element:
    """Bilinear extension of `algebra.multiply` (which acts on canonical
    labels) to general `Element`s."""
    out_terms: dict = {}
    for lbl_a, coef_a in a.terms.items():
        for lbl_b, coef_b in b.terms.items():
            prod = algebra.multiply(lbl_a, lbl_b)
            for sub_lbl, sub_coef in prod.terms.items():
                contrib = coef_a * coef_b * sub_coef
                if sub_lbl in out_terms:
                    out_terms[sub_lbl] = out_terms[sub_lbl] + contrib
                else:
                    out_terms[sub_lbl] = contrib
    out_terms = {k: v for k, v in out_terms.items() if not v.is_zero()}
    return Element(out_terms)


def _trace_dict(algebra: KAlgebra, a: Element, K: int) -> dict:
    """Bilinear extension of `algebra.trace` to general `Element`s,
    returned as a `{q-exponent: ring-coefficient}` dict truncated at
    `q^K`.  Two traces compare equal iff their dicts are equal.

    The ring coefficient is `int` for `TrivialZPlusRing` (unflavoured)
    algebras and `RElement` for flavoured algebras.  Both are
    handled uniformly: per-term contribution is `vc * v`, where `vc`
    is the LaurentPoly's int q-coefficient and `v` is the trace's
    R-coefficient (int or RElement).  The first contribution per
    `exp` initialises the slot; subsequent ones accumulate via `+`.
    """
    out: dict = {}
    for lbl, coef in a.terms.items():
        sub = algebra.trace(lbl, K=K)
        sub_coeffs = sub.coeffs  # dict-like {exponent: ring_value}
        # `coef` is the Element's coefficient: a `LaurentPoly` (`_coeffs`,
        # int q-coefficients) in the common case, or an `RLaurent`
        # (`coeffs`, `RElement` q-coefficients) when the source algebra
        # carries flavour directly in the coefficient ring (e.g. the
        # BPS-chart `SU2Nf1BpsRForm`, μ ∈ R).  Both multiply cleanly into the
        # R-valued trace.
        coef_coeffs = coef._coeffs if hasattr(coef, "_coeffs") else coef.coeffs
        for e, v in sub_coeffs.items():
            for ec, vc in coef_coeffs.items():
                exp = e + ec
                if exp > K:
                    continue
                term = vc * v
                if exp in out:
                    out[exp] = out[exp] + term
                else:
                    out[exp] = term
    def _is_zero(x):
        if isinstance(x, int):
            return x == 0
        return x.is_zero()
    return {e: v for e, v in out.items() if not _is_zero(v)}


def _rho_conjugate_coef(coef):
    """ρ acts on coefficients by the flavour-ring ⋆ (`μ^f ↦ μ^{-f}`,
    `RElement.star`) on the R-side, leaving the q-grading untouched (ρ is
    *not* the bar involution).  For a plain `LaurentPoly` (q-only, no
    flavour) this is the identity, so the common case is unchanged; for
    an `RLaurent` (flavour carried in `R`, e.g. the BPS-chart
    `SU2Nf1BpsRForm`) it conjugates each q-coefficient's `RElement`."""
    if hasattr(coef, "coeffs") and not hasattr(coef, "_coeffs"):  # RLaurent
        from zplus_ring import RLaurent
        return RLaurent(coef.ring,
                        {qe: rel.star() for qe, rel in coef.coeffs.items()})
    return coef


def _rho(algebra: KAlgebra, a: Element) -> Element:
    """ρ-semilinear extension of `algebra.rho` to general `Element`s.

    `algebra.rho(label)` may return a bare `Label` (the common case —
    flavour carried in a label slot, e.g. cone's `μ_p`) or an `Element`
    (when ρ shifts a coefficient-ring fugacity and the labels are
    flavour-free, e.g. the BPS-chart `SU2Nf1BpsRForm`, ρ μ-shifting into
    `R`).  Coefficients are conjugated by `_rho_conjugate_coef` (ρ acts
    on `R` by `μ^f ↦ μ^{-f}`); for q-only coefficients this is the
    identity, so label-slot-flavour algebras are unaffected."""
    out = Element.zero()
    for lbl, coef in a.terms.items():
        shifted = algebra.rho(lbl)
        coef = _rho_conjugate_coef(coef)
        if isinstance(shifted, Element):
            out = out + shifted * coef
        else:
            out = out + Element({shifted: coef})
    return out
