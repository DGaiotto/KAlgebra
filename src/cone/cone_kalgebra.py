"""`ConeKAlgebra` — a `KAlgebra` presented via `ConeData` + a Layer-2
residual-trace rule.

Why this exists
---------------
You generally cannot just write down a global list of generators and a
multiplication table for a `KAlgebra` — the canonical basis is infinite
and the structure constants are intricate.  But in many examples — often
by *abstracting / hard-coding the lessons that `BPSKAlgebra` computes* —
you can organize the canonical basis into **cones** of known form
(maximal pairwise-q-commuting families; q-normal-ordered monomials in a
finite set of multiplicative generators) and describe how the cone
generators multiply (the cross-products + cocycle).  `ConeKAlgebra` is
that **closed-form presentation tier**: a subclass supplies the cone
organization (a `ConeData` instance) plus the generator cross-products,
and `multiply` / `trace` are then derived universally.

Where it sits
-------------
`ConeKAlgebra(KAlgebra)` is the *closed-form* realisation tier — as
opposed to the *computed* tiers `RGKAlgebra` (RG-flow) and its subclass
`BPSKAlgebra` (BPS-quiver), from which the cone structure is frequently
distilled.  Abelianized-chart structure is a *logically independent*
axis (which chart / realisation the presentation is written in), **not**
a competing sibling: a concrete abelianized algebra can use the
cone-based multiply of `ConeKAlgebra` while organizing its basis in an
abelianized chart.

Concrete subclasses provide the algebra's defining data through:

  * a `ConeData` instance (mult-gens, q-commute cocycle, cross-products,
    cone-label bijection) — see `cone_data.py`;
  * a Layer-2 residual-trace rule `_trace_residual(seed, K)` giving the
    closed-form (Nahm-sum / character / Andrews–Gordon / …) value of
    each trace seed produced by the Layer-1 reducer.

The algebra's `multiply` and `trace` are then **inherited universally**
from `ConeKAlgebra`:

  * `multiply(a, b)` ≡ `cone_data().derived_multiply(a, b)`
    (the validated tagged-cycle reducer in `cone_data.py`).
  * `trace(a, K)` runs Layer 1 = `simplify_trace_via_cone_data(a)`
    (ρ²-cyclicity-driven reduction to a `Z[q^±]`-linear combination of
    trace seeds), then plugs in Layer 2 via `_trace_residual` on each
    seed.

The closed-form `_kalg` family (Pentagon, Heptagon, U1Hexagon, A1A2k,
…) is the natural population — typically a closed-form distillation of
what the BPS / RG presentations compute.

The specific **form of the cones can vary** (the change of basis between
PBW monomials in the mult-gens and the canonical basis differs from cone
to cone), which is why `cone_data.py` carries the composable `Cone` class —
one concrete class parameterized by a partition of its mult-gens into
monomial (identity change-of-basis), torus (invertible directions), and
character (Chebyshev / Weyl) kinds.  The partition is carried as data
(`torus_gens` / `char_gens`) and queried via `is_monomial()` /
`is_quantum_torus()` / `is_character()`; the PBW ↔ canonical map
dispatches on it.

Concrete subclasses must still implement the remaining `KAlgebra`
primitives:

    coefficient_ring, identity, rho, rho_inverse

and the two `ConeKAlgebra`-level abstracts:

    cone_data, _trace_residual

**Flavour-lift coordinate.**  The section split is exposed through
`r_label_decompose` (the single-irrep `(section, R-basis-label)`
coordinate of `KAlgebra`), which `ConeKAlgebra` **defaults** for cone
presentations: gauge-only / flavour-in-coefficients cones lift
trivially, and a subclass that instead supplies the RElement-valued
`_label_section_decompose` has its coordinate read off it automatically
(the reverse bridge — see that method's docstring).  So
`_label_section_decompose` is not required of new cone theories: a
flavour-in-a-label-slot theory (e.g. `A1D3`) supplies
`r_label_decompose` directly instead.  Its **inverse**
`r_label_compose` (`(section, R-basis-label) → label`) is the
label-producing partner — used by the flavour-changing methods
(`forget` / `lower_flavour`) and the section-factored
`KAlgebraIso.from_section_map`; its default routes through the central
embedding `embed_R`, which a peel theory may bypass with a direct
slot-write override.  `embed_R` is the central embedding
`ι : R ↪ A_𝖖` (with its faithfulness / ρ-compatibility axioms) and
backs `from_R_form`.

See `cone_data.py` for the `Cone` taxonomy (the monomial /
quantum-torus / character partition), the §"Flavour in the cone
presentation" (the five ways `G_f` interacts with cones — flavour in
labels vs coefficients vs split-out χ-index vs ad-hoc character cones),
and the universality (so-far empirical) of the Layer-1 tagged-cycle
algorithm.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from abc import abstractmethod

from kalgebra import KAlgebra, Element, Label
from cone_data import ConeData
from laurent_poly import LaurentPoly
from zplus_ring import (
    RElement, RLaurent, RPowerSeries, TrivialZPlusRing, AbelianZPlusRing,
    BasisElement,
)


__all__ = ["ConeKAlgebra"]


def _central_unit_key(R, charge):
    """The `R`-character key of the central unit `μ^charge` (`charge` =
    the U(1) shift from the `_rho_delta` table).  For an abelian flavour
    ring the charge tuple *is* the key; for `SU(2)×U(1)` the U(1) shift
    rides the U(1) factor with a trivial `SU(2)` character `χ_0`."""
    if isinstance(R, AbelianZPlusRing):
        return tuple(charge)
    # SU(2)×U(1): basis key (k, m) = (SU(2) spin index, U(1) charge);
    # the centre is the U(1) factor → χ_0 ⊗ μ^charge.
    cls_name = type(R).__name__
    if cls_name == "SU2xU1ZPlusRing":
        return (0,) + tuple(charge)
    return tuple(charge)


class ConeKAlgebra(KAlgebra):
    """`KAlgebra` defined by a `ConeData` instance + a residual-trace
    rule.  See module docstring.

    Promoted from `KAlgebra`'s optional `cone_data() -> ConeData | None`
    to a load-bearing primitive: subclasses must return a non-None
    `ConeData`.  The `multiply` and `trace` `KAlgebra` primitives are
    inherited universally; subclasses provide only the analytic
    Layer-2 trace via `_trace_residual`.
    """

    # ----- promoted from optional to load-bearing -------------------------

    @abstractmethod
    def cone_data(self) -> ConeData:
        """The `ConeData` presenting this algebra.  Must not return
        `None` (unlike the optional `KAlgebra.cone_data()`)."""

    # ----- universal multiply --------------------------------------------

    def multiply(self, a: Label, b: Label) -> Element:
        """Inherited universally: tagged-cycle reduction via the
        cone-data primitives.  See `ConeData.derived_multiply`."""
        return self.cone_data().derived_multiply(a, b)

    # ----- ρ on elements: the honest twist for flavoured cones -----------

    def rho_element(self, x: Element) -> Element:
        """ρ extended to an `Element`, the genuine algebra automorphism.

        For an **unflavoured / self-dual-coefficient** cone (`R = Z`, or a
        ring whose `⋆` is the identity and which carries no `_rho_delta`)
        this is the base label-permutation with coefficients carried
        unchanged — byte-identical to `KAlgebra.rho_element`.

        For a **flavour-in-coefficients** cone (non-trivial `R` *with* a
        `_rho_delta` table — the U(1) entries a3/a5/a7/e7 and the
        SU(2)×U(1) entries a1d4/a1d6/a1d8) the centre is non-trivially
        ρ-acted, so ρ is genuinely twisted:

            ρ(c · L_w) = ⋆(c) · μ^{δ(w)} · L_{ρ(w)}

        — apply the rep-ring duality `⋆` to the character coefficient and
        the per-letter μ^δ shift (`δ(w) = Σ_i p_i · _rho_delta[i]`).  This
        is the twist the *generated* `rho`/`rho_element` dropped, which is
        why `verify_rho_is_automorphism` failed on those entries (ρ was a
        bare cone-letter permutation that ignored the μ in the
        coefficient).  With this override the native flavoured cone is an
        honest automorphism — no flavour-in-labels rewrite needed.
        """
        R = self.coefficient_ring()
        delta = getattr(self, "_rho_delta", None)
        if isinstance(R, TrivialZPlusRing) or not delta:
            return super().rho_element(x)
        rank = getattr(self, "_L_basis_rank", getattr(R, "rank", 1))
        out: dict[Label, RLaurent] = {}
        for label, coeff in x.terms.items():
            new_label = self._canonicalized_rho_label(label)
            d = [0] * rank
            for (i, p) in label:
                di = delta.get(i)
                if di:
                    for k in range(rank):
                        d[k] += di[k] * p
            unit = R.basis_element(_central_unit_key(R, tuple(d)))
            rl = (coeff if isinstance(coeff, RLaurent)
                  else RLaurent(R, {e: RElement(R, {R.one_basis(): c})
                                    for e, c in coeff._coeffs.items()}))
            twq: dict[int, RElement] = {}
            for q_exp, r_el in rl.coeffs.items():
                shifted = r_el.star() * unit
                if not shifted.is_zero():
                    twq[q_exp] = (shifted if q_exp not in twq
                                  else twq[q_exp] + shifted)
            tw = RLaurent(R, twq)
            out[new_label] = tw if new_label not in out else out[new_label] + tw
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def _canonicalized_rho_label(self, label: Label) -> Label:
        """The ρ-image label, re-canonicalised within its cone.  The bare
        letter permutation `rho(label)` can land on a *non-canonical* word
        in a non-simplicial cone (two words, one canonical basis element);
        `multiply` canonicalises internally, so ρ must too or ρ(ab) and
        ρ(a)ρ(b) disagree on those words.  A basis-permuting ρ admits no
        q-phase here (the phase relates two normalisations of the *same*
        lattice point, a property of the word, not of ρ — validated by an
        exhaustive ρ-automorphism sweep), so it is dropped."""
        perm = self.rho(label)
        if len(perm) < 2:
            return perm
        cd = self.cone_data()
        if not hasattr(cd, "cone_of_label") or not hasattr(
            cd, "canonicalize_cone_label"
        ):
            return perm
        try:
            gens_fs, powers = cd.to_cone_label(perm)
            cone = cd.cone_of_label(perm)
            mg = cone.mult_gens() if hasattr(cone, "mult_gens") else cone
            g2, p2, _qphase = cd.canonicalize_cone_label(mg, gens_fs, powers)
            return tuple(sorted((i, p2[i]) for i in g2 if p2.get(i)))
        except Exception as exc:
            # The realisation advertises cone canonicalization but failed on
            # this label — returning the uncanonicalised `perm` is exactly
            # what this method exists to prevent, so do it loudly.
            import warnings
            warnings.warn(
                f"_canonicalized_rho_label({label!r}): cone canonicalization "
                f"failed ({type(exc).__name__}: {exc}); returning the "
                f"uncanonicalised permuted label.",
                RuntimeWarning, stacklevel=2)
            return perm

    # ρ²-orbit canonicalisation is provided by `KAlgebra` (default:
    # orbit walk with safety bound).  Subclasses with infinite ρ²-orbits
    # (e.g. `U1SquareKAlg`) MUST override
    # `_canonical_rho2_orbit_rep` with a closed-form drift-quotient
    # canonicalisation.  See `KAlgebra._canonical_rho2_orbit_rep` docstring.

    # ----- flavour-lift coordinate: the (section, R-basis-label) surface ----
    #
    # `KAlgebra` carries the *forward* bridge `_label_section_decompose`
    # (the `(section, RElement)` form) → `r_label_decompose` (the
    # single-irrep `(section, R-basis-label)` lift coordinate).  Some cone
    # theories supply the former; the override below is the *reverse*
    # bridge, so each gets a correct `r_label_decompose` for free — no
    # per-theory edit.

    def r_label_decompose(
        self, label: Label,
    ) -> "tuple[Label, BasisElement]":
        """The single-irrep flavour-lift coordinate, **defaulted for cone
        presentations** (see `r_label_decompose` on `KAlgebra` for the
        contract, and `cone_data.py` §"Flavour in the cone presentation").

        Cone theories split into exactly two label shapes:

          * **gauge-only / flavour-in-coefficients** — the label *is* the
            section and nothing is peeled, `(label, R.one_basis())`
            (Pentagon, `U1Square`, `U1Hexagon`, and the generated finite
            zoo incl. the Hexagon `FiniteA3` and the SU(2)×U(1)
            `FiniteA1D4`, whose μ/χ live in the cross-product coefficients);
          * **flavour-in-a-label-slot** — a dedicated coordinate carries the
            flavour character, `(tile, a, b, k) ↦ ((tile, a, b, 0), k)`
            (`A1D3`/`A1D5`/`A1D7`, `SU3AD`; the μ-power slot of `SU2Nf1`).

        Both shapes are recovered here from the RElement-valued
        `_label_section_decompose` **when a subclass supplies it**, by
        reading the single irrep off its coefficient — so every such cone
        theory acquires a correct `r_label_decompose` with no change.
        When a subclass instead implements `r_label_decompose` directly
        that override wins; when it implements neither, the gauge-only
        trivial lift is returned.

        The recursion guard (`_label_section_decompose` still the inherited
        `KAlgebra` default) is what keeps this *reverse* bridge from looping
        against `KAlgebra`'s *forward* bridge: it reads off only when a
        subclass genuinely overrode `_label_section_decompose`, otherwise it
        returns the trivial lift directly.

        A non-single-irrep (virtual-character) section has no single-irrep
        lift and raises — such a realisation must implement
        `r_label_decompose` itself.
        """
        cls = type(self)
        if cls._label_section_decompose is not KAlgebra._label_section_decompose:
            # Subclass supplies the old RElement form: read the single irrep
            # off it (correct for both the trivial and the peel shapes).
            sec, r_coef = self._label_section_decompose(label)
            nonzero = [(b, c) for b, c in r_coef.terms.items() if c != 0]
            if len(nonzero) == 1 and nonzero[0][1] == 1:
                return sec, nonzero[0][0]
            raise NotImplementedError(
                f"{cls.__name__}.r_label_decompose: _label_section_decompose "
                f"returned a non-single-irrep coefficient ({r_coef!r}); a "
                f"virtual-character section has no single-irrep lift "
                f"coordinate — implement r_label_decompose directly."
            )
        # End-state default: gauge-only cones lift trivially; the
        # flavour-in-a-label-slot theories MUST override.
        return label, self.coefficient_ring().one_basis()

    # ----- Layer-2 trace primitive (per subclass) ------------------------

    @abstractmethod
    def _trace_residual(self, seed_label: Label, K: int) -> RPowerSeries:
        """Closed-form value of the trace at a single Layer-1 residual
        seed, as an `RPowerSeries` over `coefficient_ring()` to order `K`.

        Seeds passed in are **canonical ρ²-orbit representatives**
        produced by Layer 1 (`simplify_trace_via_cone_data`, which now
        composes the tagged-cycle algorithm with a ρ²-orbit
        canonicalisation pass).  Concretely:

          * the identity label = `cone_data().from_cone_label(frozenset(), {})`;
          * one canonical single-mult-gen label per ρ²-orbit on
            mult-gens (minimum-by-Python-sort label in each orbit).

        ρ²-invariance of trace (`Tr(ρ²(x)) = Tr(x)`) is enforced **in
        code** at Layer 1: the subclass cannot accidentally violate
        ρ²-cyclicity here because it never receives two ρ²-related
        seeds separately.  E.g. Pentagon's five single-mult-gen seeds
        `(i, 1, 0)` for `i ∈ ℤ/5` are folded into the single canonical
        representative `(0, 1, 0)` before this method is called.
        """

    # ----- universal trace (Layer 1 + Layer 2) ---------------------------

    def trace(self, a: Label, K: int = 20) -> RPowerSeries:
        """Inherited universally: Layer-1 (tagged-cycle reduction +
        ρ²-orbit canonicalisation, both inside
        `simplify_trace_via_cone_data`) followed by Layer-2 closed-form
        plug-in via `_trace_residual`.

        The widening `inner_K = K - emin` absorbs negative q-shifts
        injected by the Layer-1 recursion (cocycle phases during the
        ρ²-cyclicity slide), so the Layer-2 power series has enough
        precision after multiplication and truncation back to `K`.
        """
        cd = self.cone_data()
        simplified: Element = cd.simplify_trace_via_cone_data(self, a)

        R = self.coefficient_ring()

        # Layer-1 seed coefficients are either `LaurentPoly` (when R is
        # `TrivialZPlusRing`) or `RLaurent[R]` (when R is a non-trivial
        # flavour / character ring).  Normalise both into `RLaurent[R]`
        # exponent dicts for the lift.
        def _q_exponents(c_poly) -> "dict[int, ...]":
            if isinstance(c_poly, LaurentPoly):
                return dict(c_poly._coeffs)
            if isinstance(c_poly, RLaurent):
                return dict(c_poly.coeffs)
            raise TypeError(
                f"ConeKAlgebra.trace: unsupported seed-coefficient type "
                f"{type(c_poly).__name__}"
            )

        # Widen K by the most-negative q-power present in any seed coeff.
        emin = 0
        for c_poly in simplified.terms.values():
            for e in _q_exponents(c_poly).keys():
                if e < emin:
                    emin = e
        inner_K = K - emin

        # Accumulate `Σ_seed c_seed(q) · Tr(seed)(q)` into a single
        # RPowerSeries on the R-side.  Seeds are already
        # ρ²-canonicalised at the Layer-1 boundary.
        result_coeffs: dict = {}
        for seed_label, c_poly in simplified.terms.items():
            seed_trace: RPowerSeries = self._trace_residual(
                seed_label, inner_K
            )
            if isinstance(c_poly, RLaurent):
                if c_poly.ring != R:
                    raise ValueError(
                        f"ConeKAlgebra.trace: seed-coefficient RLaurent ring "
                        f"mismatch (cone-data R = {c_poly.ring}, "
                        f"algebra R = {R})"
                    )
                c_rl = c_poly
            else:
                c_rl = RLaurent(R, dict(c_poly._coeffs))
            product = seed_trace * c_rl
            for e, c in product.coeffs.items():
                if e <= K:
                    if e in result_coeffs:
                        result_coeffs[e] = result_coeffs[e] + c
                    else:
                        result_coeffs[e] = c

        return RPowerSeries(R, result_coeffs, K)
