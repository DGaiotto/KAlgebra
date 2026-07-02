"""`TensorKAlgebra`: graded tensor product of two `KAlgebra`s over Z[q^±].

Contract:

    TensorKAlgebra(A, B)  over  Z[q, q^{-1}]

    label              :  pair (α, β) ∈ basis_set(A) × basis_set(B)
    coefficient_ring   :  the tensor of the *non-trivial* factor rings --
                          `TrivialZPlusRing` when both factors are over
                          Z[q±], `R_A` / `R_B` when one factor is
                          flavoured, `TensorZPlusRing(R_A, R_B)` when
                          both are.
    identity           :  (1_A, 1_B)
    multiply           :  factor-wise tensor:
                          multiply((α₁,β₁),(α₂,β₂)) =
                              Σ_{γ_A,γ_B} c^A_{γ_A}(q) · c^B_{γ_B}(q)
                                          · (γ_A, γ_B)
    rho                :  (ρ_A α, ρ_B β)
    rho_inverse        :  (ρ_A⁻¹ α, ρ_B⁻¹ β)
    trace((α,β))       :  trace_A(α) · trace_B(β)   (RPowerSeries product,
                          each factor lifted into the common ring)
    _label_section_decompose :  factor-wise

No cross-pairing -- the two factors literally commute over Z[q±].
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element, KAlgebra
from laurent_poly import LaurentPoly
from zplus_ring import (
    TrivialZPlusRing, TensorZPlusRing,
    identity_hom, unit_hom, tensor_inclusion_hom,
)


class TensorKAlgebra(KAlgebra):
    """Graded tensor `A ⊗ B` of two `KAlgebra`s over Z[q±].

    Constructor:

        TensorKAlgebra(A: KAlgebra, B: KAlgebra)

    All four flavour combinations are supported: both factors over
    `TrivialZPlusRing` (Z[q±], the direct-product path), one flavoured
    factor (the common ring is that factor's `R`), or both flavoured
    (the common ring is `TensorZPlusRing(R_A, R_B)`); each factor's
    trace is lifted into the common ring by its flavour-growing hom.
    """

    def __init__(self, A: KAlgebra, B: KAlgebra) -> None:
        self._A = A
        self._B = B
        R_A = A.coefficient_ring()
        R_B = B.coefficient_ring()
        self._A_triv = isinstance(R_A, TrivialZPlusRing)
        self._B_triv = isinstance(R_B, TrivialZPlusRing)
        # Coefficient ring = the product of the *non-trivial* factor rings
        # (a trivial `Z` factor contributes nothing — `Z ⊗ R = R`, no spurious
        # `TrivialZPlusRing ⊗ R`).  The two factors' traces are lifted into
        # this common ring by the flavour-growing homs (`unit_hom` /
        # `tensor_inclusion_hom`).
        if self._A_triv and self._B_triv:
            self._R = TrivialZPlusRing()
            self._phi_A = self._phi_B = None          # trivial path: direct product
        elif self._B_triv:                            # only A flavoured ⇒ R = R_A
            self._R = R_A
            self._phi_A = identity_hom(R_A)
            self._phi_B = unit_hom(R_A)
        elif self._A_triv:                            # only B flavoured ⇒ R = R_B
            self._R = R_B
            self._phi_A = unit_hom(R_B)
            self._phi_B = identity_hom(R_B)
        else:                                         # both flavoured ⇒ R = R_A ⊗ R_B
            self._R = TensorZPlusRing(R_A, R_B)
            self._phi_A = tensor_inclusion_hom(self._R, 0)
            self._phi_B = tensor_inclusion_hom(self._R, 1)

    # ----- KAlgebra contract ----------------------------------------------

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return (self._A.identity(), self._B.identity())

    def multiply(self, a, b):
        a_A, a_B = a
        b_A, b_B = b
        prod_A = self._A.multiply(a_A, b_A)
        prod_B = self._B.multiply(a_B, b_B)
        out: dict[tuple, LaurentPoly] = {}
        for lbl_A, c_A in prod_A.terms.items():
            if c_A.is_zero():
                continue
            for lbl_B, c_B in prod_B.terms.items():
                if c_B.is_zero():
                    continue
                c = c_A * c_B
                if c.is_zero():
                    continue
                key = (lbl_A, lbl_B)
                out[key] = out.get(key, LaurentPoly.zero()) + c
        return Element({k: v for k, v in out.items() if not v.is_zero()})

    def rho(self, a):
        a_A, a_B = a
        return (self._A.rho(a_A), self._B.rho(a_B))

    def rho_inverse(self, a):
        a_A, a_B = a
        return (self._A.rho_inverse(a_A), self._B.rho_inverse(a_B))

    def trace(self, a, K=20):
        """`trace((α, β)) = trace_A(α) · trace_B(β)`, as an RPowerSeries over the
        common coefficient ring.  When both factors are over `Z` the product is
        direct (the historical path); when a factor is flavoured, each factor's
        trace is first lifted into the common ring by its flavour-growing hom
        (`unit_hom` for a `Z` factor, `tensor_inclusion_hom` when both are
        flavoured), so e.g. `Tr((α, μ-torus^m)) = μ^m · Tr_A(α)`."""
        a_A, a_B = a
        tr_A = self._A.trace(a_A, K)
        tr_B = self._B.trace(a_B, K)
        if self._phi_A is None:                       # both over Z: direct product
            return tr_A * tr_B
        return (self._phi_A.apply_RPowerSeries(tr_A)
                * self._phi_B.apply_RPowerSeries(tr_B))

    def _label_section_decompose(self, label):
        """Factor-wise section decomposition.  Both factors over `Z`: trivial
        section `(label, one)` for *any* label shape (the whole label is its own
        section — the unpacking is skipped, so a flat RGKAlgebra label passed by
        `_section_split` still works).  With a flavoured factor: combine each
        factor's section coefficient lifted into the common ring by its growing
        hom (labels are genuine `(a_A, a_B)` 2-tuples in that case)."""
        if self._phi_A is None:                       # both over Z
            return label, self._R.one()
        a_A, a_B = label
        sec_A, r_A = self._A._label_section_decompose(a_A)
        sec_B, r_B = self._B._label_section_decompose(a_B)
        coef = (self._phi_A.apply_RElement(r_A)
                * self._phi_B.apply_RElement(r_B))
        return (sec_A, sec_B), coef

    def r_label_decompose(self, label):
        """The flavour-lift coordinate, factor-wise.  Both factors over `Z`:
        the lift is trivial (section = label, R-basis-label = `()`).  With a
        flavoured factor the section is `(sec_A, sec_B)` and the R-basis-label
        is the surviving factor's irrep (`Z ⊗ R = R` keeps only the flavoured
        side) or the pair `(r_A, r_B)` when both factors are flavoured (the
        `TensorZPlusRing` basis element).

        A trivial-ring factor decomposes as `(label, ())` directly (it need not
        implement the optional `r_label_decompose` — a `Z`-factor without one
        still works).  Both factors over `Z`: the lift is trivial for *any*
        label shape (the unpacking is skipped, mirroring
        `_label_section_decompose`)."""
        if self._phi_A is None:                       # both over Z
            return label, ()
        a_A, a_B = label
        sec_A, r_A = (a_A, ()) if self._A_triv else self._A.r_label_decompose(a_A)
        sec_B, r_B = (a_B, ()) if self._B_triv else self._B.r_label_decompose(a_B)
        section = (sec_A, sec_B)
        if self._B_triv:                              # R = R_A
            return section, r_A
        if self._A_triv:                              # R = R_B
            return section, r_B
        return section, (r_A, r_B)                    # R = R_A ⊗ R_B

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`."""
        if self._phi_A is None:                       # both over Z (trivial lift)
            return section
        sec_A, sec_B = section
        if self._B_triv:                              # R = R_A
            a_A = sec_A if self._A_triv else self._A.r_label_compose(sec_A, r_basis_label)
            a_B = sec_B
        elif self._A_triv:                            # R = R_B
            a_A = sec_A
            a_B = sec_B if self._B_triv else self._B.r_label_compose(sec_B, r_basis_label)
        else:                                         # R = R_A ⊗ R_B
            r_A, r_B = r_basis_label
            a_A = self._A.r_label_compose(sec_A, r_A)
            a_B = self._B.r_label_compose(sec_B, r_B)
        return (a_A, a_B)

    # ----- introspection --------------------------------------------------

    @property
    def factor_A(self) -> KAlgebra:
        return self._A

    @property
    def factor_B(self) -> KAlgebra:
        return self._B
