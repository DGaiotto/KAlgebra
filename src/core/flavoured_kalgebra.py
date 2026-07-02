"""`AddFlavourKAlgebra` — adjoin a spectator flavour symmetry to any KAlgebra.

**An alternative label-coordinate encoding of flavour.**  This wrapper
encodes flavour as an extra *label coordinate* `f`.  The other encoding
carries flavour in the *coefficient ring* instead ("free over `R(G_f)`"),
grown functorially by `base_change` through a flavour-growing ring hom
(`zplus_ring.unit_hom` / `tensor_inclusion_hom`) and forgotten by
`base_change(augmentation)` (`KAlgebra.forget()`); that coefficient-ring
encoding is preferred for new code.  The two present the **same** flavoured
algebra under `(L_a, f) ↔ χ_f·L_a` (see `KAlgebra.add_flavour`).  This
wrapper is retained because some matter-dressing `RGKAlgebra` flows use it
as their auxiliary.

Given any `KAlgebra` `B` and any `ZPlusRing` `R` (the representation ring of
the flavour symmetry), this wrapper produces a new `KAlgebra`

    add_flavour(B, R)  =  B  with a spectator flavour whose rep ring is R

A label is a pair `(base_label, f)` with `f` a **basis element of `R`** — a
flavour irrep / character.  The abelian case `R = AbelianZPlusRing(n)` is the
`U(1)^n` flavour (`f ∈ Z^n`); a non-abelian `R` (`SU2ZPlusRing`,
`SU3ZPlusRing`, …) gives genuine non-abelian flavour, products Clebsch-Gordan'd.

The coefficient ring grows:

    B unflavoured  (TrivialZPlusRing)   ->  R
    B flavoured    (R_B non-trivial)    ->  TensorZPlusRing(R_B, R)

The whole construction is **driven by the `ZPlusRing` interface** of `R`:

    identity flavour   =  R.one_basis()
    multiply flavour   =  R.multiply_basis(fa, fb)   (Clebsch-Gordan; multi-term
                                                       for non-abelian R)
    rho on flavour     =  R.star_basis(f)            (rep dual: `-f` abelian,
                                                       id for self-dual SU(2))
    trace character    =  R.basis_element(f)         (`μ^f` abelian, `χ_f` else)

Relation to `base_change` (the preferred path)
----------------------------------------------
`KAlgebra.base_change(phi)` is a **coefficient-only pushforward**: it keeps the
same labels and the same Z-form `multiply`, and only re-homes the trace's
R-coefficients through `phi`.  This wrapper instead adds the flavour irrep as
**new label coordinates**.  Though they look like genuinely different
constructions, they present the **same** flavoured algebra:
`base_change` through a flavour-growing hom (`unit_hom` / `tensor_inclusion_hom`,
`zplus_ring`) makes the algebra free over `R(G_f)` with the *unflavoured* basis,
and the label-coordinate basis here maps to it by `(L_a, f) ↦ χ_f·L_a`
(trace-compatible).  The coefficient-ring encoding is the preferred one for
new code (it composes cleanly with `forget = base_change(augmentation)` and
`lower_flavour = base_change(restriction)`); this label-coordinate wrapper is
the alternative, retained as the auxiliary of some matter-dressing RG flows.

Z-form = tensor with a flavour rep ring
---------------------------------------
In the Z-form this *is* `B ⊗ R(flavour)` with the flavour factor central
(q-commuting, group-like, ρ-dualized).  But unlike a bare `TensorKAlgebra`, the
result is presented as a genuine *flavoured* KAlgebra
(`_label_section_decompose` / `embed_R` fold the flavour charge onto the
character `R.basis_element(f)` in the R-form view).

Primary use
-----------
The auxiliary of a matter-dressing `RGKAlgebra` flow: the IR
pure-gauge algebra `B` acquires the matter flavour symmetry, the RG grading
`Γ_RG` is read straight off the flavour-charge label coordinate, and the
matter–gauge coupling lives entirely in `S_RG = ∏_R ∏_{w∈R} E_𝖖(μ_R v^w)`.

Generality
----------
Any `ZPlusRing` factor — abelian `U(1)^n` (`AbelianZPlusRing`), non-abelian
`SU(2)`/`SU(3)`/… (`SU2ZPlusRing`, `SU3ZPlusRing`, `SO3ZPlusRing`, …), or a
tensor/product of these.  `add_flavour(n)` (an int) is shorthand for
`AbelianZPlusRing(n)`.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element, Label
from zplus_ring import (
    ZPlusRing, TrivialZPlusRing, AbelianZPlusRing,
    RElement, RPowerSeries,
)


__all__ = ["AddFlavourKAlgebra", "AddAbelianFlavourKAlgebra"]


class AddFlavourKAlgebra(KAlgebra):
    """`B` with a trivially-added spectator flavour whose rep ring is `R`.

    Labels are pairs `(base_label, f)` with `f` a basis element of `R` (a
    flavour irrep / character).  See the module docstring.

    `R` is given to the constructor as `flavour`: either an int `n`
    (shorthand for `AbelianZPlusRing(n)`) or any `ZPlusRing`.
    """

    def __init__(self, base: KAlgebra, flavour) -> None:
        if isinstance(flavour, bool) or not isinstance(flavour, (int, ZPlusRing)):
            raise TypeError(
                "add_flavour: `flavour` must be an int (U(1)^n shorthand) or a "
                f"ZPlusRing; got {type(flavour).__name__}"
            )
        if isinstance(flavour, int):
            if flavour < 1:
                raise ValueError(
                    f"add_flavour: integer rank must be >= 1, got {flavour}"
                )
            flavour = AbelianZPlusRing(rank=flavour)
        if isinstance(flavour, TrivialZPlusRing):
            raise ValueError(
                "add_flavour: a TrivialZPlusRing flavour adds nothing; pass an "
                "int rank >= 1 or a non-trivial ZPlusRing"
            )
        self._base = base
        self._flav: ZPlusRing = flavour
        self._zero_f = flavour.one_basis()

        R_B = base.coefficient_ring()
        self._base_trivial = isinstance(R_B, TrivialZPlusRing)
        if self._base_trivial:
            self._R: ZPlusRing = self._flav
        else:
            from tensor_zplus_ring import TensorZPlusRing
            self._R = TensorZPlusRing(R_B, self._flav)

    # ----- introspection --------------------------------------------------

    @property
    def base(self) -> KAlgebra:
        return self._base

    @property
    def flavour_ring(self) -> ZPlusRing:
        return self._flav

    @property
    def n_flavour(self) -> int:
        """Abelian rank — only when the flavour ring is `AbelianZPlusRing`."""
        if isinstance(self._flav, AbelianZPlusRing):
            return self._flav.rank
        raise AttributeError(
            "n_flavour is defined only for abelian flavour; use `flavour_ring`"
        )

    # ----- R-side combine: pair a base-R character with the flavour character -

    def _combine_r(self, r_B: RElement, f) -> RElement:
        """Lift a base-R character `r_B` (over `R_B`) tensored with the flavour
        character `R.basis_element(f)` into an `RElement` over `coefficient_ring()`.

        * unflavoured base: `r_B = z·1 ∈ Z`  ↦  `z·χ_f`.
        * flavoured base:   `r_B = Σ c_b·b`  ↦  `Σ c_b·(b, f)` in `R_B ⊗ R`.
        """
        if self._base_trivial:
            z = r_B.terms.get((), 0)
            return RElement(self._R, {f: z}) if z else self._R.zero()
        return RElement(self._R, {(b, f): c for b, c in r_B.terms.items()})

    # ----- KAlgebra contract ----------------------------------------------

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> Label:
        return (self._base.identity(), self._zero_f)

    def multiply(self, a: Label, b: Label) -> Element:
        """`(b_a, f_a) · (b_b, f_b)`: base multiply ⊗ flavour Clebsch-Gordan.

        The flavour reps decompose via `R.multiply_basis(f_a, f_b)` — a single
        term `f_a+f_b` for abelian `R`, a sum of irreps (with multiplicities)
        for non-abelian `R`."""
        (ba, fa), (bb, fb) = a, b
        base_prod = self._base.multiply(ba, bb)
        flav = self._flav.multiply_basis(fa, fb)   # {f_c: multiplicity}
        result = Element.zero()
        for bc, lp in base_prod.terms.items():
            if lp.is_zero():
                continue
            for fc, m in flav.items():
                result = result + Element({(bc, fc): lp}) * m
        return result

    def rho(self, a: Label) -> Label:
        (ba, fa) = a
        return (self._base.rho(ba), self._flav.star_basis(fa))

    def rho_inverse(self, a: Label) -> Label:
        (ba, fa) = a
        return (self._base.rho_inverse(ba), self._flav.star_basis(fa))

    def rho_squared_is_identity(self) -> bool:
        # ρ²((b, f)) = (ρ²_base(b), ⋆²f) = (ρ²_base(b), f) since ⋆ is an
        # involution (rep double-dual); flavour is fixed by ρ².
        return self._base.rho_squared_is_identity()

    def trace(self, a: Label, K: int = 20) -> RPowerSeries:
        """`Tr(L_{(b, f)}) = χ_f · Tr_base(L_b)`.

        The flavour directions are central, so they surface as the flavour
        character `R.basis_element(f)` multiplying the base trace."""
        (ba, fa) = a
        base_tr = self._base.trace(ba, K)
        out: dict[int, RElement] = {}
        for q_exp, rc in base_tr.coeffs.items():
            combined = self._combine_r(rc, fa)
            if not combined.is_zero():
                out[q_exp] = combined
        return RPowerSeries(self._R, out, K)

    def _label_section_decompose(
        self, label: Label,
    ) -> "tuple[Label, RElement]":
        """Fold the flavour charge (and the base's own flavour content) onto
        the flavour-trivial section `(base_section, R.one_basis())`."""
        (ba, fa) = label
        base_sec, r_B = self._base._label_section_decompose(ba)
        section = (base_sec, self._zero_f)
        r_coeff = self._combine_r(r_B, fa)
        return section, r_coeff

    def r_label_decompose(self, label):
        """The single-irrep flavour-lift coordinate `(section, r_basis_label)`:
        delegate to the base's lift coordinate and pair it with the added
        flavour irrep `f`.  Section `(base_section, 1_flav)`; R-basis-label `f`
        (unflavoured base, `R = R_flav`) or the tensor key `(base_irrep, f)`
        (flavoured base, `R = R_B ⊗ R_flav`)."""
        (ba, fa) = label
        base_sec, b_basis = self._base.r_label_decompose(ba)
        section = (base_sec, self._zero_f)
        if self._base_trivial:
            return section, fa
        return section, (b_basis, fa)

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`: rebuild `(base_label, f)` from the
        flavour-trivial section + the (possibly tensor) R-basis-label.  The
        added flavour is central — a direct pairing, no `embed_R` round-trip."""
        (base_sec, _zero) = section
        if self._base_trivial:
            return (base_sec, r_basis_label)
        (b_basis, f) = r_basis_label
        return (self._base.r_label_compose(base_sec, b_basis), f)

    def embed_R(self, r: RElement) -> Element:
        """`ι : R ↪ A`.  Unflavoured base: `χ_f ↦ L_{(1_base, f)}`.
        Flavoured base: `(b_B, f) ↦` (base central image of `b_B`) carried at
        flavour `f`."""
        if not isinstance(r, RElement) or r.ring != self._R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()"
            )
        out = Element.zero()
        if self._base_trivial:
            base_id = self._base.identity()
            for f, c in r.terms.items():
                out = out + Element.basis((base_id, f)) * c
            return out
        R_B = self._base.coefficient_ring()
        for (b_B, f), c in r.terms.items():
            base_elt = self._base.embed_R(RElement(R_B, {b_B: 1}))
            for blabel, lp in base_elt.terms.items():
                out = out + Element({(blabel, f): lp}) * c
        return out

    def __repr__(self) -> str:
        return f"AddFlavourKAlgebra({self._base!r}, {self._flav!r})"


# Backward-compatible alias: the construction is no longer abelian-only, but the
# old name still resolves (and still works for the abelian / int-rank case).
AddAbelianFlavourKAlgebra = AddFlavourKAlgebra
