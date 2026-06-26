"""`SU2Nf1KAlgebra` — `A_𝖖[SU(2) + N_f = 1]` as a standalone
`ConeKAlgebra`.

Mirrors `PureSU2KAlg` (pure-SU(2) `ConeKAlgebra`) with U(1)_F flavour:

  * mult-gens: H_n (H-tower) + w_1 (Wilson fundamental).
  * cones: rank-2 pairs `C_a = {H_a, H_{a+1}}` + Wilson cone `{w_1}`.
  * coefficient_ring: `AbelianZPlusRing(rank=1)` (μ-flavour fugacity).
  * ρ-action: `ρ(H_n) = μ^{−1} · H_{n−3}` (index shift −3 + μ shift).
  * multiply: literal-word reducer in `su2_nf1_h_multiply.multiply_native`,
    axiom-derived from `h_mul_h` cyclicity + Nf=1 Clebsch (verified
    against `SUN_Nf(2, 1).algebra` for gaps 0..9 H-H and multi-letter
    + Wilson + mixed inputs).
  * trace: cyclicity-bridge formulas in `su2_nf1_h_trace` (TBD —
    next phase, analogous to pure-SU(2) m-anchor system).
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from cone_kalgebra import ConeKAlgebra
from zplus_ring import AbelianZPlusRing, RPowerSeries
from su2_nf1_cone_data import SU2Nf1ConeData, _native_to_psu2nf1, _psu2nf1_to_native


class SU2Nf1KAlgebra(ConeKAlgebra):
    """`A_𝖖[SU(2) + N_f = 1]` as a standalone `ConeKAlgebra` over
    `AbelianZPlusRing(rank=1)` (U(1)_F flavour).
    """

    def __init__(self) -> None:
        self._R = AbelianZPlusRing(rank=1)
        self._cone_data_inst = SU2Nf1ConeData()

    # -- KAlgebra primitives -------------------------------------------

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return ((), 0)

    def cone_data(self):
        return self._cone_data_inst

    def multiply(self, a, b):
        """Axiom-derived multiplication via literal-ray-word reducer
        in `su2_nf1_h_multiply.multiply_native`.

        Structure constants come from `h_mul_h` (W_1-walk cyclicity
        recursion, verified vs `SUN_Nf(2, 1).algebra` for gaps 0..9)
        plus Nf=1 ε-corrected w_1↔H Clebsch.
        """
        from su2_nf1_h_multiply import multiply_native
        return multiply_native(a, b)

    def rho(self, label):
        """`ρ(H_n) = μ^{−1} · H_{n−3}`; ρ trivial on Wilson cone.

        ρ is a bar-twisted automorphism: it acts on coefficients via
        `μ → μ⁻¹` (σ on the flavour kernel direction (0,0,1) sends it to
        (0,0,−1), as verified directly in canonical `BPSKAlgebra`).  So
        for a label with pre-existing μ-power `mu_p`, the new μ-power
        is `−mu_p − total_exp` (bar flips the sign of `mu_p`, and each
        H-letter contributes an extra μ⁻¹ from `ρ(H_n) = μ⁻¹·H_{n−3}`).

        Consequence: `ρ²(H_n) = H_{n−6}` (μ⁰, NOT μ⁻²) — the two μ⁻¹s
        from successive ρ applications cancel under the bar twist.
        """
        h_factors, mu_p = label
        if not h_factors:
            return (h_factors, -int(mu_p))
        first = h_factors[0]
        if isinstance(first[0], tuple) and first[0][0] == 'W':
            return (h_factors, -int(mu_p))       # Wilson H-content trivial, bar on μ.
        new_factors = tuple(sorted(
            ((n - 3, exp) for (n, exp) in h_factors),
            key=lambda x: x[0],
        ))
        total_exp = sum(exp for (n, exp) in h_factors)
        return (new_factors, -int(mu_p) - total_exp)

    def rho_inverse(self, label):
        """Inverse of `rho`: H-index +3 with the same bar-twisted μ rule.

        Because the bar twist is its own inverse and the H-shift sign
        flips between ρ and ρ⁻¹, the μ rule is identical:
        `new_mu_p = −mu_p − total_exp`.  Composition `ρ ∘ ρ⁻¹ = id` is
        then automatic (bar²=id and the H-shifts ±3 cancel).
        """
        h_factors, mu_p = label
        if not h_factors:
            return (h_factors, -int(mu_p))
        first = h_factors[0]
        if isinstance(first[0], tuple) and first[0][0] == 'W':
            return (h_factors, -int(mu_p))
        new_factors = tuple(sorted(
            ((n + 3, exp) for (n, exp) in h_factors),
            key=lambda x: x[0],
        ))
        total_exp = sum(exp for (n, exp) in h_factors)
        return (new_factors, -int(mu_p) - total_exp)

    # ----- Element-level ρ with μ-canonicalisation ---------------------
    #
    # `rho(label)` may return a label with `μ_pow ≠ 0`, whereas `multiply`
    # outputs always have `μ_pow = 0` with μ-content in the R-coefficient.
    # To make `verify_rho_is_automorphism` pass (Element equality requires
    # matching label representations), `rho_element` canonicalises each
    # output label to `μ_pow = 0` and lifts the μ-shift into the coef.

    def rho_element(self, x):
        """`ρ` linearly extended to Element, with output canonicalised so
        every label has `μ_pow = 0` and any μ-power sits in the
        `RLaurent` coefficient.  Matches the convention used by
        `multiply` so Z-form automorphism `ρ(a·b) = ρ(a)·ρ(b)` holds as
        Element equality.

        Bar-twist on R: ρ is a bar-twisted automorphism — the μ-content
        of each `RLaurent` coefficient must be sent `μ → μ⁻¹` as ρ
        propagates through.  Implemented by flipping the sign of every
        μ-monomial index in the coefficient.
        """
        from kalgebra import Element
        from zplus_ring import RLaurent, RElement
        def _bar_mu(c) -> RLaurent:
            if not hasattr(c, "coeffs"):   # plain Z-form LaurentPoly
                c = RLaurent(self._R, {e: RElement(self._R, {(0,): v})
                                       for e, v in c._coeffs.items()})
            out = {}
            for q_e, r_e in c.coeffs.items():
                new_terms = {}
                for k, v in r_e.terms.items():
                    new_key = (-k[0],) if isinstance(k, tuple) else (-k,)
                    new_terms[new_key] = new_terms.get(new_key, 0) + v
                out[q_e] = RElement(self._R, new_terms)
            return RLaurent(self._R, out)
        out: dict = {}
        for a, c in x.terms.items():
            new_h_factors, new_mu_p = self.rho(a)
            canonical = (new_h_factors, 0)
            # Apply bar-twist to coefficient's μ-content.
            barred = _bar_mu(c)
            # Lift any extra μ-shift from `new_mu_p` into the coefficient.
            if new_mu_p != 0:
                mu_coef = self._R.basis_element((int(new_mu_p),))
                lifted = RLaurent(self._R, {e: v * mu_coef
                                              for e, v in barred.coeffs.items()})
            else:
                lifted = barred
            if canonical in out:
                out[canonical] = out[canonical] + lifted
            else:
                out[canonical] = lifted
        return Element({k: v for k, v in out.items() if not v.is_zero()})

    def rho_inverse_element(self, x):
        """Same canonicalisation + bar-twist for ρ⁻¹."""
        from kalgebra import Element
        from zplus_ring import RLaurent, RElement
        def _bar_mu(c) -> RLaurent:
            if not hasattr(c, "coeffs"):   # plain Z-form LaurentPoly
                c = RLaurent(self._R, {e: RElement(self._R, {(0,): v})
                                       for e, v in c._coeffs.items()})
            out = {}
            for q_e, r_e in c.coeffs.items():
                new_terms = {}
                for k, v in r_e.terms.items():
                    new_key = (-k[0],) if isinstance(k, tuple) else (-k,)
                    new_terms[new_key] = new_terms.get(new_key, 0) + v
                out[q_e] = RElement(self._R, new_terms)
            return RLaurent(self._R, out)
        out: dict = {}
        for a, c in x.terms.items():
            new_h_factors, new_mu_p = self.rho_inverse(a)
            canonical = (new_h_factors, 0)
            barred = _bar_mu(c)
            if new_mu_p != 0:
                mu_coef = self._R.basis_element((int(new_mu_p),))
                lifted = RLaurent(self._R, {e: v * mu_coef
                                              for e, v in barred.coeffs.items()})
            else:
                lifted = barred
            if canonical in out:
                out[canonical] = out[canonical] + lifted
            else:
                out[canonical] = lifted
        return Element({k: v for k, v in out.items() if not v.is_zero()})

    def r_label_decompose(self, label):
        """The single-irrep flavour-lift coordinate (replaces the retired
        `_label_section_decompose`): peel the U(1) flavour weight `μ^{μ_power}`
        (the R-basis-label `(μ_power,)`) off the gauge section
        `(h_factors, 0)`."""
        h_factors, mu_p = label
        return (h_factors, 0), (int(mu_p),)

    def r_label_compose(self, section, r_basis_label):
        """Inverse of `r_label_decompose`: write the μ-power back into the
        flavour slot.  `SU2Nf1KAlgebra` supplies no `embed_R`, so the default
        (`embed_R`-based) compose cannot rebuild a non-trivial-μ label — this
        is the genuine label-producer (a direct slot write)."""
        h_factors, _ = section
        (mu_p,) = r_basis_label
        return (h_factors, int(mu_p))

    # -- Trace via cyclicity reduction to Tr(W_n) (Schur F) ----------
    #
    # Note: the cone-data Layer-1 simplification path
    # (`simplify_trace_via_cone_data`) is not adequate for SU(2)+Nf=1
    # (it collapses distinct Wilson characters χ_e to the same cone
    # representative).  Trace is therefore implemented directly from
    # the native seed label, bypassing Layer 1, with `_trace_residual`
    # retained only as an ABC contract stub.

    def _trace_residual(self, seed_label, K: int):
        """ABC contract stub — unused.  See `trace` below for the actual
        cyclicity-based trace.
        """
        from su2_nf1_h_trace import trace_seed
        return trace_seed(seed_label, K)

    def trace(self, label, K: int = 20):
        """`Tr(L_label) ∈ R((q))` via cyclicity reduction to Tr(W_n).

        Wilson m=0 traces come from the SU(2)+Nf=1 Schur F(v, μ) closed
        form (`su2_nf1_h_trace.tr_W`).  Higher-m anchors are obtained
        from ρ²-twisted cyclicity bridges reducing to Tr(W_n) (see
        `su2_nf1_h_trace.trace_seed` / cyclicity solvers).
        """
        from su2_nf1_h_trace import trace_seed
        return trace_seed(label, K)
