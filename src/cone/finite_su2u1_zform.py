"""Z-form (API-compliant) wrapper for the **su2u1-flavoured** finite
standalones (a1d4 / a1d6 / a1d8).

The su2u1 analogue of `finite_u1_zform.FiniteU1ZKAlgebra`, and the
documented Plan-18 fix for the su2u1 ρ defect.  The frozen su2u1
standalones carry **bare section** labels (flavour absorbed into the
RLaurent coefficient) and implement `rho` as a pure section permutation,
so ρ has nowhere to put the flavour element `r` it must pick up
(`ρ(L_s) = r·L_{Ps}`).  Consequence: `verify_rho_is_automorphism` fails
on the generators whose ρ-image carries a nonzero `r` (a1d8: 78/120
generators carry a U(1) shift; the SU(2) part is ρ-invariant because
χ_n is Weyl/self-dual).

Fix (the author's framing: *ρ permutes canonical labels; in the flavoured
case it maps a section to `(section, r)`*).  Give the labels the explicit
`(free, (word, k))` structure of the RKAlgebra free⊗label encoding,
**combining the two existing Z-form patterns**:

* the **su2-free** part (`finite_a1d3_zform`): `free_ring = R(SU(2))`,
  ⋆ = identity on χ_n, the SU(2) characters fuse via the engine;
* the **u1-label** part (`finite_u1_zform`): `label_ring = R(U(1))`, the
  U(1) charge lives explicitly in the section `k`, and
  `ρ((w, k)) = (P·w, δ(w) − k)` (the `−k` is ⋆ on the U(1) centre, the
  `δ` is the per-letter `RHO_DELTA` shift).

The native SU(2)×U(1) coefficient ring (basis `(n, m)` = SU(2) irrep ×
U(1) charge) is split as `R(SU(2)) ⊗ R(U(1))` via `(n, m) ↦ (n, (m,))`,
so the native cross-tables' SU(2) content routes to `free` and their
U(1) charge routes into the section `k`.

Requires the native standalone to export `_rho_delta` (the U(1) shift,
1-component) and `_L_basis_rank = 1`.  a1d8 shipped without `_rho_delta`
(like e7); it was computed by a probe in the source repository
(the e7 recipe in the native U(1) frame) and is inlined in
`finite_a1d8_kalg.py` (78 of the 120 generators carry a nonzero shift), so
all three su2u1 standalones meet the requirement (checked 2026-09-26: a1d4's
table is all zeros, a1d6 carries 25 nonzero shifts, a1d8 78).

The flavour-lift coordinate `r_label_decompose` / `r_label_compose` (added
2026-09-26, the design record) puts the SU(2) weight and the U(1) charge in the key,
`L_{(w, (word, k))} = χ_w·μ^k · L_{(0, (word, 0))}`; the wrapper is identified
with `A1DevenKAlg(k)` by a `KAlgebraIso`
(`a1deven_seeds.kalgebra_iso`).
"""
from __future__ import annotations

from kalgebra import KAlgebra, Element, ElementOverR
from laurent_poly import LaurentPoly
from zplus_ring import (
    SU2ZPlusRing, AbelianZPlusRing, RElement, RLaurent, RPowerSeries,
)
from tensor_zplus_ring import TensorZPlusRing


class FiniteSU2U1ZKAlgebra(KAlgebra):
    """Generic Z-form wrapper over a su2u1-flavoured frozen standalone.

    `free_ring = R(SU(2))`, `label_ring = R(U(1))`; the section is
    `(word, k)` with `k ∈ Z` (the U(1) charge).  ρ is an honest
    label permutation: `(w_su2, (word, k)) ↦ (w_su2, (P·word, δ(word) − k))`
    (⋆ fixes the SU(2) character, inverts+shifts the U(1) charge).
    """

    def __init__(self, native):
        if not hasattr(native, "_rho_delta") or not getattr(
                native, "_rho_delta", None):
            raise NotImplementedError(
                f"{type(native).__name__} has no usable _rho_delta (the U(1) "
                f"shift table) — compute it (e7 recipe in the native U(1) "
                f"frame) and inline it before wrapping (honest-fail).")
        self._native = native
        self._rank = getattr(native, "_L_basis_rank", 1)
        if self._rank != 1:
            raise NotImplementedError(
                f"su2u1 Z-form expects a single U(1) axis (_L_basis_rank=1), "
                f"got {self._rank}")
        self._free = SU2ZPlusRing()
        self._lab = AbelianZPlusRing(rank=1)

    # -------- helpers --------

    def _delta(self, word):
        d = 0
        for (i, p) in word:
            di = self._native._rho_delta.get(i)
            if di is None:
                continue
            d += di[0] * p
        return d

    def _canon_word(self, word):
        """Canonicalize a cone word (non-simplicial cones admit relations:
        the letterwise ρ-image can land on a non-canonical word).  Mirrors
        `finite_u1_zform._canon_word`; honest basis-permuting ρ admits no
        q-phase here."""
        if len(word) < 2:
            return word
        cd = self._native.cone_data()
        try:
            gens_fs, powers = cd.to_cone_label(word)
            cone = cd.cone_of_label(word)
            g2, p2, _qph = cd.canonicalize_cone_label(
                cone.mult_gens() if hasattr(cone, "mult_gens") else cone,
                gens_fs, powers)
            return tuple(sorted((i, p2[i]) for i in g2 if p2.get(i)))
        except Exception:
            return word

    # -------- section engine --------

    def free_ring(self):
        return self._free

    def label_ring(self):
        return self._lab

    def section_identity(self):
        return (self._native.identity(), 0)

    def _lift_native_relem(self, re):
        """Native `RElement` over SU(2)×U(1) (basis `(n, m)`) ↦ `RElement`
        over `R(SU(2)) ⊗ R(U(1))` (basis `(n, (m,))`)."""
        R = self.coefficient_ring()
        return RElement(R, {(n, (m,)): c for (n, m), c in re.terms.items()})

    def _lift_native_rlaurent(self, rl):
        R = self.coefficient_ring()
        return RLaurent(R, {q: self._lift_native_relem(c)
                            for q, c in rl.coeffs.items()})

    def section_multiply(self, s1, s2) -> ElementOverR:
        (w1, k1), (w2, k2) = s1, s2
        base = self._native.multiply(w1, w2)
        k12 = k1 + k2
        R = self.coefficient_ring()
        out = {}
        for w, c in base.terms.items():
            lc = self._lift_native_rlaurent(c) if isinstance(c, RLaurent) else c
            out[(w, k12)] = lc
        return ElementOverR(R, out)

    def shift_section(self, s, label_basis):
        w, k = s
        return (w, k + label_basis[0])

    def section_rho(self, s):
        w, k = s
        return (self._canon_word(self._native.rho(w)), self._delta(w) - k)

    def section_rho_inverse(self, s):
        u, m = s
        w = self._canon_word(self._native.rho_inverse(u))
        return (w, self._delta(w) - m)

    # ---- KAlgebra contract: free⊗label fusion (from RKAlgebra/u1 z-form) ----

    def coefficient_ring(self):
        cached = getattr(self, "_rk_coeff_ring", None)
        if cached is not None:
            return cached
        R = TensorZPlusRing(self._free, self._lab)
        self._rk_coeff_ring = R
        return R

    def _split(self, full_basis):
        free_b, lab_b = full_basis
        return free_b, lab_b

    def _lift_free_elem(self, e: RElement) -> RElement:
        R = self.coefficient_ring()
        lab_one = self._lab.one_basis()
        return RElement(R, {(fb, lab_one): n for fb, n in e.terms.items()})

    def _section_of(self, label_basis):
        if label_basis is None:
            return self.section_identity()
        return self.shift_section(self.section_identity(), label_basis)

    def _pack_label(self, free_char, section):
        return (free_char, section)

    def _unpack_label(self, label):
        return label

    def identity(self):
        return self._pack_label(self._free.one_basis(), self.section_identity())

    def multiply(self, a, b) -> Element:
        w1, s1 = self._unpack_label(a)
        w2, s2 = self._unpack_label(b)
        Rf = self._free
        section = self.section_multiply(s1, s2)              # ElementOverR
        cw = self._lift_free_elem(
            Rf.basis_element(w1) * Rf.basis_element(w2)
        )                                                    # RElement over full R
        out: dict = {}
        for u, rl in section.terms.items():
            for qpow, relem in rl.coeffs.items():
                fused = cw * relem                           # RElement over full R
                for full_b, n in fused.terms.items():
                    if n == 0:
                        continue
                    fb, lb = self._split(full_b)
                    u2 = self.shift_section(u, lb)
                    lab = self._pack_label(fb, u2)
                    term = LaurentPoly({qpow: n})
                    out[lab] = term if lab not in out else out[lab] + term
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def rho(self, label):
        w, s = self._unpack_label(label)
        return self._pack_label(self._free.star_basis(w), self.section_rho(s))

    def rho_inverse(self, label):
        w, s = self._unpack_label(label)
        return self._pack_label(self._free.star_basis(w),
                                self.section_rho_inverse(s))

    def _label_section_decompose(self, label):
        w, s = self._unpack_label(label)
        r_coeff = self._lift_free_elem(self._free.basis_element(w))
        return self._pack_label(self._free.one_basis(), s), r_coeff

    # -------- the flavour-lift coordinate --

    def r_label_decompose(self, label):
        """The single-irrep flavour-lift coordinate `(section, R-basis key)`,
        `L_label = χ_key · L_section`: the section is the SU(2)-singlet,
        charge-0 label `(0, (word, 0))`, and the key is `(w, (k,))`, the SU(2)
        weight with the U(1) charge — `L_{(w, (word, k))} = χ_w·μ^k ·
        L_{(0, (word, 0))}`.

        Added 2026-09-26 for the `KAlgebraIso` onto
        `A1DevenKAlg(k)` (`a1deven_seeds.kalgebra_iso`),
        whose `verify_maps_section_to_section_1drep` reads this coordinate on
        both algebras.  `_label_section_decompose` keeps its own override
        above — the free⊗label R-form, whose section keeps the U(1) charge —
        so `section_decompose` / `to_R_form` are unchanged; the consumers of
        this coordinate are the flavour-change wrappers (`forget`,
        `base_change`, `lower_flavour`, which prefer it to
        `_label_section_decompose`) and `from_R_form`, through
        `r_label_compose`."""
        w, (word, k) = self._unpack_label(label)
        return self._pack_label(self._free.one_basis(), (word, 0)), (w, (k,))

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`: the character's charge is added to
        the section's (so a section carrying a charge, as the sections of
        `_label_section_decompose` do, composes consistently too)."""
        _one, (word, k0) = self._unpack_label(section)
        w, (k,) = r_basis_label
        return self._pack_label(w, (word, k0 + k))

    def embed_R(self, r: RElement) -> Element:
        R = self.coefficient_ring()
        if not isinstance(r, RElement) or r.ring != R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()")
        out: dict = {}
        for full_b, n in r.terms.items():
            if n == 0:
                continue
            fb, lb = self._split(full_b)
            lab = self._pack_label(fb, self._section_of(lb))
            term = LaurentPoly({0: n})
            out[lab] = term if lab not in out else out[lab] + term
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def trace(self, a, K: int = 20):
        """`Tr((w, (word, k))) = χ_w·μ^k · Tr_native(word)` — R-linear over
        the centre.  Until 2026-09-26 the SU(2) character `χ_w` of the label
        was dropped (the u1 wrapper's trace, whose free ring is trivial,
        copied without it), so every label with `w ≠ 0` got the trace of its
        `w = 0` section; found by the `KAlgebraIso` onto `A1DevenKAlg(k)`
        (`a1deven_seeds.kalgebra_iso`), whose trace check
        fails on `χ_1` before the fix and passes after it."""
        w_free, (word, k) = self._unpack_label(a)
        base = self._native.trace(word, K)
        R = self.coefficient_ring()
        lifted = RPowerSeries(
            R, {e: self._lift_native_relem(c) for e, c in base.coeffs.items()},
            base.K)
        if k == 0 and w_free == self._free.one_basis():
            return lifted
        return lifted * RElement(R, {(w_free, (k,)): 1})
