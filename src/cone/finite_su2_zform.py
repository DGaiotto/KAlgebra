"""Z-form (API-compliant) wrapper for the **SU(2)-flavoured** finite
standalones (a1d3, a1d5, a1d7).

The SU(2) member of the family of Z-form wrappers, beside
`finite_u1_zform.FiniteU1ZKAlgebra(native)` (u1-flavoured) and
`finite_su2u1_zform.FiniteSU2U1ZKAlgebra(native)` (su2u1-flavoured), and the
generalisation of `finite_a1d3_zform.FiniteA1D3ZKAlgebra`, which is this class
over `FiniteA1D3KAlgebra`.

the design record: a flavoured `KAlgebra` returns **Z-valued** `multiply`
(`LaurentPoly`, flavour in labels).  The generated SU(2) standalones instead
return `RLaurent` coefficients over `R(SU(2))` on bare cone-word labels
(flavour in coefficients, the R-form).  This wrapper moves the flavour into the
labels.  A canonical label is `(w, u)`, `w` an SU(2) highest weight and `u` a
native cone word, and

    L_{(w, u)}  =  χ_w · M_u .

The algebra is **fully free** over `R = R(SU(2))`: there is no torus part (no
`R_lab` factor, unlike the u1 and su2u1 wrappers), so `coefficient_ring()` is
`R(SU(2))` itself.  ρ acts on the centre by `⋆`, which fixes every SU(2)
character (they are self-dual), and SU(2) has no nontrivial one-dimensional
representation, so ρ picks up no flavour factor (no `_rho_delta` table is
needed): the section ρ is the native letter permutation, canonicalised on a
non-simplicial cone as in the u1 / su2u1 wrappers (the three SU(2)
standalones' cones are simplicial, so there it is the identity).

The trace is R-linear over the centre: `Tr((w, u)) = χ_w · Tr_native(u)`, with
`Tr_native` the standalone's own trace (for a1d3 through
`a1d3_seeds`, for a1d5 / a1d7 the `a1d5_layer2` /
`a1d7_layer2` closed forms).

Validated against the native R-form through `to_R_form`
(the suite in the source repository at a1d3, the suite in the source repository
at a1d5 / a1d7), and identified with the family class `A1DoddConeKAlg(k)` by a
`KAlgebraIso` (`a1dodd_seeds.kalgebra_iso`).
"""
from __future__ import annotations

from kalgebra import KAlgebra, Element, ElementOverR
from laurent_poly import LaurentPoly
from zplus_ring import RElement, SU2ZPlusRing


class FiniteSU2ZKAlgebra(KAlgebra):
    """Generic Z-form wrapper over an SU(2)-flavoured frozen standalone.

    Labels `(w, u)`: `w` the SU(2) highest weight (a basis key of
    `R(SU(2))`), `u` the native cone word; `L_{(w, u)} = χ_w · M_u`.  Fusion
    copied from `RKAlgebra` and specialised to the fully-free case (the design record
    step 1: decouple `RKAlgebra` ahead of its redesign).
    """

    def __init__(self, native):
        if not isinstance(native.coefficient_ring(), SU2ZPlusRing):
            raise NotImplementedError(
                f"{type(native).__name__} is not SU(2)-flavoured (its "
                f"coefficient ring is {native.coefficient_ring()!r}); the "
                f"u1 and su2u1 standalones have their own Z-form wrappers "
                f"(finite_u1_zform, finite_su2u1_zform)")
        self._native = native

    # -------- the section engine (the only realisation data) --------

    def free_ring(self):
        return self._native.coefficient_ring()        # R(SU(2))

    def section_identity(self):
        return self._native.identity()                # ()

    def section_multiply(self, s1, s2) -> ElementOverR:
        """The native cone cross-tables, reinterpreted as the R-form
        section product (RLaurent on section labels)."""
        base = self._native.multiply(s1, s2)
        return ElementOverR(self.coefficient_ring(), dict(base.terms))

    def _canon_word(self, word):
        """The canonical cone word of `word` (a non-simplicial cone admits
        two words for one element, so a letterwise ρ-image can be
        non-canonical).  Mirrors `finite_u1_zform._canon_word`: the
        q-phase between two words of one lattice point is a property of
        the words, not of ρ, and is dropped."""
        if len(word) < 2:
            return word
        cd = self._native.cone_data()
        gens, powers = cd.to_cone_label(word)
        cone = cd.cone_of_label(word)
        g2, p2, _qph = cd.canonicalize_cone_label(cone.mult_gens(), gens,
                                                  powers)
        return cd.from_cone_label(frozenset(g for g in g2 if p2.get(g)),
                                  {g: p for g, p in p2.items() if p})

    def section_rho(self, s):
        return self._canon_word(self._native.rho(s))

    def section_rho_inverse(self, s):
        return self._canon_word(self._native.rho_inverse(s))

    # ---- KAlgebra contract: fully-free fusion (copied from RKAlgebra) ----

    def coefficient_ring(self):
        # Fully free over R_free = R(SU(2)); no R_lab torsor.
        return self.free_ring()

    def identity(self):
        return (self.free_ring().one_basis(), self.section_identity())

    def multiply(self, a, b) -> Element:
        w1, s1 = a
        w2, s2 = b
        Rf = self.free_ring()
        section = self.section_multiply(s1, s2)             # ElementOverR over Rf
        cw = Rf.basis_element(w1) * Rf.basis_element(w2)    # RElement over Rf
        out: dict = {}
        for u, rl in section.terms.items():                 # rl : RLaurent
            for qpow, relem in rl.coeffs.items():           # relem : RElement
                fused = cw * relem                          # RElement over Rf
                for fb, n in fused.terms.items():           # fb : R_free char
                    if n == 0:
                        continue
                    lab = (fb, u)
                    term = LaurentPoly({qpow: n})
                    out[lab] = term if lab not in out else out[lab] + term
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def rho(self, label):
        w, s = label
        return (self.free_ring().star_basis(w), self.section_rho(s))

    def rho_inverse(self, label):
        w, s = label
        return (self.free_ring().star_basis(w), self.section_rho_inverse(s))

    def trace(self, a, K: int = 20):
        """`Tr((w, u)) = χ_w · Tr_native(u)` through `𝖖^K` — the trace is
        R-linear over the centre, and the native standalone serves
        `Tr_native` on its bare cone words."""
        w, s = a
        base = self._native.trace(s, K)
        R = self.coefficient_ring()
        if w == R.one_basis():
            return base
        return base * R.basis_element(w)

    def r_label_decompose(self, label):
        """`(w, s) ↦ ((free_one, s), w)` — single-irrep flavour-lift
        coordinate: section `(free_one, s)`, single SU(2) irrep key `w`
        (the fully-free case carries no R_lab charge, so the section is
        just the trivial-character dressing of `s`)."""
        w, s = label
        return (self.free_ring().one_basis(), s), w

    def r_label_compose(self, section, r_basis_label):
        _free_one, s = section
        return (r_basis_label, s)

    def embed_R(self, r: RElement) -> Element:
        """Central embedding `χ_w ↦ L_{(w, s₀)}`, extended Z-linearly."""
        R = self.coefficient_ring()
        if not isinstance(r, RElement) or r.ring != R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()"
            )
        s0 = self.section_identity()
        out: dict = {}
        for fb, n in r.terms.items():
            if n == 0:
                continue
            lab = (fb, s0)
            term = LaurentPoly({0: n})
            out[lab] = term if lab not in out else out[lab] + term
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def __repr__(self) -> str:
        return f"FiniteSU2ZKAlgebra({type(self._native).__name__})"
