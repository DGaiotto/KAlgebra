"""Z-form (API-compliant) wrapper for the **u1-flavoured** finite
standalones (a3, a5, a7; the polygon entries when frozen).

The frozen u1 standalones implement `rho(label)` as a pure permutation
of cone letters, dropping the flavour bookkeeping: the axiom requires
ρ to invert the central flavour character (`ρ(μ·x) = μ⁻¹·ρ(x)`), and
each letter additionally carries a μ-shift `δ(i)` under ρ
(`ρ(L_w) = μ^{δ(w)}·L_{Pw}`, `δ(w) = Σ p·δ(i)` — the standalones store
this correctly in `_rho_delta` but only use it in the bespoke
`rho_R_element`, never on the contract surface).  Consequence: ~73–80
verifier violations per entry on the universal surface
(the audit / the design record).

Fix (pattern, as `finite_a1d3_zform.py` for the SU(2)
case): wrap the native engine in `RKAlgebra` with **sections carrying
the μ-charge explicitly** — section = `(word, k)` with `k ∈ Z^rank` —
so flavour lives in labels and ρ is an honest label permutation:

    ρ((w, k))   = (P·w,    δ(w) − k)
    ρ⁻¹((u, m)) = (P⁻¹·u,  δ(P⁻¹·u) − m)

(roundtrip: ρ⁻¹ρ(w,k) = (w, δ(w) − (δ(w) − k)) = (w, k)).

`free_ring` is trivial (u1 characters are group-like ⇒ all label-ring);
the engine's `shift_section` absorbs μ-powers from the native
cross-tables into the `k` component.
"""
from __future__ import annotations

from kalgebra import KAlgebra, Element, ElementOverR
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing, RElement, RLaurent, RPowerSeries
from tensor_zplus_ring import TensorZPlusRing


class FiniteU1ZKAlgebra(KAlgebra):
    """Generic Z-form wrapper over a u1-flavoured frozen standalone.

    the design record step 1: decoupled from `RKAlgebra` — the general (R_lab-aware)
    free/label fusion is **copied in** below so `RKAlgebra` can be
    redesigned without constraining this wrapper.  Here `R_free` is
    trivial and `R_lab = R(U(1)^rank)` (the native flavour ring), so the
    fusion routes the u1 character through `shift_section` into the `k`
    component of the `(word, k)` section.
    """

    def __init__(self, native):
        if not hasattr(native, "_rho_delta"):
            raise NotImplementedError(
                f"{type(native).__name__} predates the rho-delta export; "
                f"regenerate the standalone via finite_kalgebras.regen "
                f"before wrapping (honest-fail, not a silent wrong rho).")
        self._native = native
        self._rank = native._L_basis_rank
        self._free = TrivialZPlusRing()

    # -------- helpers --------

    def _delta(self, word):
        d = [0] * self._rank
        for (i, p) in word:
            di = self._native._rho_delta.get(i)
            if di is None:
                continue
            for j in range(self._rank):
                d[j] += di[j] * p
        return tuple(d)

    @staticmethod
    def _kadd(k1, k2, sign=1):
        return tuple(a + sign * b for a, b in zip(k1, k2))

    def _canon_word(self, word):
        """Canonicalize a cone word (non-simplicial cones admit
        relations: two words, one algebra element — e.g. a7's
        L44·L46 = L_{(43,47)}).  The letterwise ρ-image can land on a
        non-canonical word; ρ must return the canonical one.  An
        honest basis-permuting ρ admits no q-phase here."""
        if len(word) < 2:
            return word
        cd = self._native.cone_data()
        gens_fs, powers = cd.to_cone_label(word)
        cone = cd.cone_of_label(word)
        g2, p2, _qph = cd.canonicalize_cone_label(cone.mult_gens()
                                                  if hasattr(cone, 'mult_gens')
                                                  else cone, gens_fs, powers)
        # The q-phase relates the two NORMALIZATIONS of the same lattice
        # point (word vs canonical word); it is a property of the word,
        # not of ρ.  The canonical ρ is basis→basis with the charge-
        # derived μ^δ only, so the label map is the canonical word with
        # the phase dropped — a choice ADJUDICATED, not assumed: the
        # exhaustive pairwise ρ-automorphism sweep over all letters
        # (tests) fails by stray q-factors if this is wrong.
        return tuple(sorted((i, p2[i]) for i in g2 if p2.get(i)))

    def _zero_k(self):
        return (0,) * self._rank

    # -------- RKAlgebra section engine --------

    def free_ring(self):
        return self._free

    def label_ring(self):
        return self._native.coefficient_ring()

    def section_identity(self):
        return (self._native.identity(), self._zero_k())

    # -- native-ring -> engine tensor-ring lifts (chars (k,) -> ((), (k,)))

    def _lift_relem(self, re):
        R = self.coefficient_ring()
        fb = self.free_ring().one_basis()
        return RElement(R, {(fb, ch): n for ch, n in re.terms.items()})

    def _lift_rlaurent(self, rl):
        R = self.coefficient_ring()
        return RLaurent(R, {q: self._lift_relem(c)
                            for q, c in rl.coeffs.items()})

    def section_multiply(self, s1, s2) -> ElementOverR:
        (w1, k1), (w2, k2) = s1, s2
        base = self._native.multiply(w1, w2)
        k12 = self._kadd(k1, k2)
        R = self.coefficient_ring()
        out = {}
        for w, c in base.terms.items():
            lc = self._lift_rlaurent(c) if isinstance(c, RLaurent) else c
            out[(w, k12)] = lc
        return ElementOverR(R, out)

    def shift_section(self, s, label_basis):
        w, k = s
        return (w, self._kadd(k, tuple(label_basis)))

    def section_rho(self, s):
        w, k = s
        return (self._canon_word(self._native.rho(w)),
                self._kadd(self._delta(w), k, sign=-1))

    def section_rho_inverse(self, s):
        u, m = s
        w = self._canon_word(self._native.rho_inverse(u))
        return (w, self._kadd(self._delta(w), m, sign=-1))

    # ---- KAlgebra contract: general fusion (copied from RKAlgebra) ----

    def _label_is_trivial(self) -> bool:
        return isinstance(self.label_ring(), TrivialZPlusRing)

    def coefficient_ring(self):
        cached = getattr(self, "_rk_coeff_ring", None)
        if cached is not None:
            return cached
        R = (self.free_ring() if self._label_is_trivial()
             else TensorZPlusRing(self.free_ring(), self.label_ring()))
        self._rk_coeff_ring = R
        return R

    def _split(self, full_basis):
        """`full R` basis char ↦ `(free_basis, label_basis_or_None)`."""
        if self._label_is_trivial():
            return full_basis, None
        free_b, lab_b = full_basis
        return free_b, lab_b

    def _lift_free_elem(self, e: RElement) -> RElement:
        """An `RElement` over `R_free` ↦ the same element of full `R`."""
        if self._label_is_trivial():
            return e
        R = self.coefficient_ring()
        lab_one = self.label_ring().one_basis()
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
        return self._pack_label(self.free_ring().one_basis(),
                                self.section_identity())

    def multiply(self, a, b) -> Element:
        w1, s1 = self._unpack_label(a)
        w2, s2 = self._unpack_label(b)
        Rf = self.free_ring()
        section = self.section_multiply(s1, s2)            # ElementOverR
        cw = self._lift_free_elem(
            Rf.basis_element(w1) * Rf.basis_element(w2)
        )                                                   # RElement over full R
        out: dict = {}
        for u, rl in section.terms.items():                 # rl : RLaurent
            for qpow, relem in rl.coeffs.items():           # relem : RElement
                fused = cw * relem                          # RElement over full R
                for full_b, n in fused.terms.items():
                    if n == 0:
                        continue
                    fb, lb = self._split(full_b)
                    u2 = u if lb is None else self.shift_section(u, lb)
                    lab = self._pack_label(fb, u2)
                    term = LaurentPoly({qpow: n})
                    out[lab] = term if lab not in out else out[lab] + term
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    def rho(self, label):
        w, s = self._unpack_label(label)
        return self._pack_label(self.free_ring().star_basis(w),
                                self.section_rho(s))

    def rho_inverse(self, label):
        w, s = self._unpack_label(label)
        return self._pack_label(self.free_ring().star_basis(w),
                                self.section_rho_inverse(s))

    def _label_section_decompose(self, label):
        w, s = self._unpack_label(label)
        free_one = self.free_ring().one_basis()
        r_coeff = self._lift_free_elem(self.free_ring().basis_element(w))
        return self._pack_label(free_one, s), r_coeff

    # -------- the flavour-lift coordinate --

    def r_label_decompose(self, label):
        """The single-irrep flavour-lift coordinate `(section, R-basis key)`,
        `L_label = χ_key · L_section`: the section is the label with its U(1)
        charge removed, `(free_one, (word, 0))`, and the key is the full
        character `(w_free, k)` — so `L_{(w, (word, k))} = μ^k ·
        L_{(one, (word, 0))}` (`shift_section` adds the charge; `embed_R`
        puts `μ^k` on the identity word).

        Added 2026-09-26 for the `KAlgebraIso` onto
        `ungauge_u1a1aodd(k)` (`aodd_seeds.kalgebra_iso`),
        whose `verify_maps_section_to_section_1drep` reads this coordinate on
        both algebras.  `_label_section_decompose` keeps its own override
        above — the free⊗label R-form, whose section keeps the charge — so
        `section_decompose` / `to_R_form` are unchanged; the consumers of
        this coordinate are the flavour-change wrappers (`forget`,
        `base_change`, `lower_flavour`, which prefer it to
        `_label_section_decompose`) and `from_R_form`, through
        `r_label_compose`."""
        w, (word, k) = self._unpack_label(label)
        one = self.free_ring().one_basis()
        if self._label_is_trivial():
            return self._pack_label(one, (word, k)), w
        return self._pack_label(one, (word, self._zero_k())), (w, tuple(k))

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`: the character's charge is added to
        the section's (so a section carrying a charge, as the sections of
        `_label_section_decompose` do, composes consistently too)."""
        _one, (word, k0) = self._unpack_label(section)
        if self._label_is_trivial():
            return self._pack_label(r_basis_label, (word, k0))
        w, k = r_basis_label
        return self._pack_label(w, (word, self._kadd(k0, tuple(k))))

    def embed_R(self, r: RElement) -> Element:
        R = self.coefficient_ring()
        if not isinstance(r, RElement) or r.ring != R:
            raise TypeError(
                "embed_R: argument must be an RElement over coefficient_ring()"
            )
        out: dict = {}
        for full_b, n in r.terms.items():
            if n == 0:
                continue
            fb, lb = self._split(full_b)
            lab = self._pack_label(fb, self._section_of(lb))
            term = LaurentPoly({0: n})
            out[lab] = term if lab not in out else out[lab] + term
        return Element({l: c for l, c in out.items() if not c.is_zero()})

    # -------- trace (section-level, from the native zoo Layer 2) -----

    def trace(self, a, K: int = 20):
        _w_free, (word, k) = self._unpack_label(a)
        base = self._native.trace(word, K)
        R = self.coefficient_ring()
        fb = self.free_ring().one_basis()
        lifted = RPowerSeries(
            R, {e: self._lift_relem(c) for e, c in base.coeffs.items()},
            base.K)
        if all(x == 0 for x in k):
            return lifted
        mu_k = RElement(R, {(fb, tuple(k)): 1})
        return lifted * mu_k
