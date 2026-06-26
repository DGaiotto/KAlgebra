"""`SU2Nf2ConeKAlgebra` — `A_𝖖[SU(2)+N_f=2]` as a standalone, spine-free
`ConeKAlgebra` over the Spin(4) = SU(2)_L × SU(2)_R flavour ring.

Mirrors `SU2Nf1KAlgebra` (N_f=1) lifted U(1)_F → Spin(4):

  * mult-gens: H-tower `H_n` + Wilson fundamental `w_1`.
  * coefficient_ring: `TensorZPlusRing(SU2, SU2)` (Spin(4) characters).
  * native label: `(h_factors, (wL, wR))` (H/Wilson cone monomial + Cartan
    flavour weight).
  * multiply: closed-form spine-free reducer (`su2_nf2_h_multiply`), the
    Spin(4) lift of N_f=1's W_1-walk `h_mul_h` + matter Clebsch; validated
    exhaustively against the BPS oracle (`su2_nf2_kalgebra`).
  * ρ: `ρ(H_n)=H_{n-2}` (shift 4-N_f), flavour weight negated; w_1 fixed.
  * trace: Wilson/identity via the closed-form Schur F (`su2_nf2_h_trace`);
    magnetic seeds via the orthonormality bootstrap (in progress).

NO BPS / RG / quantum-torus engine on any path.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from cone_kalgebra import ConeKAlgebra
from zplus_ring import RPowerSeries, SU2ZPlusRing
from tensor_zplus_ring import TensorZPlusRing
from su2_nf2_cone_data import SU2Nf2ConeData
from su2_nf2_h_multiply import multiply_native
from pure_su2_h_wilson import is_wilson_label


class SU2Nf2ConeKAlgebra(ConeKAlgebra):
    """`A_𝖖[SU(2)+N_f=2]` standalone `ConeKAlgebra` over Spin(4)."""

    def __init__(self) -> None:
        self._R = TensorZPlusRing(SU2ZPlusRing(), SU2ZPlusRing())
        self._cone = SU2Nf2ConeData()

    # -- KAlgebra primitives -------------------------------------------

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return ((), (0, 0))

    def cone_data(self):
        return self._cone

    def multiply(self, a, b):
        return multiply_native(a, b)

    # ρ(H_n) = H_{n-2}; w_1 fixed; flavour Cartan weight negated.
    def _rho_shift(self, label, shift):
        h_factors, (wL, wR) = label
        nw = (-wL, -wR)
        if not h_factors:
            return ((), nw)
        first = h_factors[0]
        if isinstance(first[0], tuple) and first[0][0] == 'W':
            return (h_factors, nw)               # Wilson H-content ρ-fixed
        new = tuple(sorted((n + shift, exp) for (n, exp) in h_factors))
        return (new, nw)

    def rho(self, label):
        return self._rho_shift(label, -2)

    def rho_inverse(self, label):
        return self._rho_shift(label, +2)

    def r_label_decompose(self, label):
        """Peel the Spin(4) flavour weight off the gauge section.  Returns
        `((h_factors, (0,0)), (wL, wR))` — the section is the flavour-neutral
        gauge cone monomial, the R-basis label is the Cartan weight."""
        h_factors, w = label
        return (h_factors, (0, 0)), tuple(w)

    def r_label_compose(self, section, r_basis_label):
        h_factors, _ = section
        return (h_factors, tuple(r_basis_label))

    # -- trace ---------------------------------------------------------

    def _trace_residual(self, seed_label, K):
        """ABC stub — see `trace`."""
        return self.trace(seed_label, K)

    def trace(self, label, K: int = 12):
        """Spine-free trace over Spin(4), total on every label.

        Identity / Wilson: closed-form Schur F (`tr_W`).  Magnetic gauge
        monomials: the character-basis cyclicity+orthonormality bootstrap
        (`su2_nf2_h_trace_bootstrap`, seeded by Tr(1)+Wilson), anchors solved
        in the Spin(4) irrep basis and validated against the BPS oracle.

        **Flavour-charged single labels** `(hf, (wL,wR))` with `(wL,wR)≠0`.  A
        bare charged line is **one Cartan section** of a flavour-Weyl orbit —
        not a Spin(4) class function — so its bare trace is Weyl-COVARIANT,
        centred on `(wL,wR)`.  The flavour Weyl `(μ_L→μ_L⁻¹, μ_R→μ_R⁻¹)` is a
        symmetry of the canonical basis, so the canonical, well-defined
        physical quantity is the **Weyl-invariant section**: recenter by
        `(−wL,−wR)`.  Recentering removes the flavour weight entirely, leaving
        the trace of the neutral gauge seed `(hf, (0,0))` in the Spin(4) irrep
        basis (verified identical to the engine-backed `SU2Nf2KAlgebra.trace`
        of the corresponding tropical label, which recenters the same way —
        e.g. `Tr(γ_1=(1,0,1,0)) = Tr(H_0) = −χ_{(1,0)}q − χ_{(3,0)}q³`).  So a
        charged label routes to its neutral-seed trace.

        (The flavour content of a charged label *as it sits inside a product*
        — the covariant, weight-shifted form — is handled R-linearly by
        `inner_product` / the bilinear `multiply`; this method returns the
        section, the intrinsic class-function content.)
        """
        from su2_nf2_h_trace_bootstrap import trace_label_spin4
        h_factors, (wL, wR) = label
        # Charged labels: recenter to the Weyl-invariant section == the neutral
        # gauge-seed trace (same gauge content, flavour weight removed).
        return trace_label_spin4((h_factors, (0, 0)), K)

    def inner_product(self, a, b, K: int = 12):
        """`I_{a,b} = Tr(ρ(L_a)·L_b) = δ_{a,b} + O(q)` over Spin(4), computed
        directly from the product decomposition + solved anchors (so the
        flavour content of the matter terms is handled correctly).  Verified:
        δ at q⁰, no negative-q, for the magnetic generators."""
        from su2_nf2_h_trace_bootstrap import inner_product_spin4
        return inner_product_spin4(a, b, K)


__all__ = ["SU2Nf2ConeKAlgebra"]
