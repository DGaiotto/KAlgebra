"""`SU2Nf3ConeKAlgebra` — `A_𝖖[SU(2)+N_f=3]` as a standalone, spine-free
`ConeKAlgebra` over the SU(4) flavour ring.

The SU(4) sibling of `SU2Nf2ConeKAlgebra` (Spin(4)) / `SU2Nf1KAlgebra`
(U(1)_F):

  * mult-gens: H-tower `H_n` + Wilson fundamental `w_1`.
  * coefficient_ring: `SU4ZPlusRing` (= R(SU(4)); matter is the SO(6)=6=∧²4).
  * native label: `(h_factors, (m1, m2, m3))` (H/Wilson cone monomial + SU(4)
    Cartan flavour weight, Dynkin/ω-basis).
  * multiply: the spine-free **literal-word reducer** (`su2_nf3_h_multiply`),
    the SU(4) sibling of N_f=2 — TOTAL on every magnetic level.  Inputs (canonical
    cone monomials) expand to literal H/Wilson words via the inverse Gaussian
    q-binomial; only adjacent PAIRS are multiplied (`su2_nf3_h_gap_k.h_mul_h`,
    all gaps, always magnetic ≤ 2; ε_n = 4/4̄ by parity); the word is reduced
    (matter-aware swap/bubble: `H_xH_{x+1}=q²H_{x+1}H_x+(1−q²)·1`) to canonical
    single-cone form and read back to canonical labels by the forward Gaussian.
    Validated vs the BPS oracle (incl. magnetic-3+ inputs, distant cones) and by
    the (a·b)·c==a·(b·c) associativity + bar sweep.
  * ρ: `ρ(H_n)=H_{n-1}` (shift 4−N_f=−1), flavour weight negated (star, 4↔4̄);
    w_1 fixed.
  * trace: Wilson/identity via the closed-form Schur F (`su2_nf3_h_trace`);
    magnetic seeds via the **SU(4) character-basis cyclicity+orthonormality
    bootstrap** (`su2_nf3_h_trace_bootstrap`).  Validated vs the oracle:
    Tr(H_n) exact through q⁵, orthonormality I_{a,b}=δ+O(q).

NO BPS / RG / quantum-torus engine on any path.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cone_kalgebra import ConeKAlgebra
from zplus_ring import SU4ZPlusRing
from su2_nf3_cone_data import SU2Nf3ConeData
from su2_nf3_h_multiply import multiply_native, _seed_of


class SU2Nf3ConeKAlgebra(ConeKAlgebra):
    """`A_𝖖[SU(2)+N_f=3]` standalone `ConeKAlgebra` over SU(4)."""

    def __init__(self) -> None:
        self._R = SU4ZPlusRing()
        self._cone = SU2Nf3ConeData()

    # -- KAlgebra primitives -------------------------------------------

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return ((), (0, 0, 0))

    def cone_data(self):
        return self._cone

    def multiply(self, a, b):
        return multiply_native(a, b)

    # ρ(H_n) = H_{n-1}; w_1 fixed; flavour Cartan weight negated (star).
    def _rho_shift(self, label, shift):
        h_factors, w = label
        nw = tuple(-x for x in w)
        if not h_factors:
            return ((), nw)
        first = h_factors[0]
        if isinstance(first[0], tuple) and first[0][0] == 'W':
            return (h_factors, nw)                # Wilson H-content ρ-fixed
        new = tuple(sorted((n + shift, exp) for (n, exp) in h_factors))
        return (new, nw)

    def rho(self, label):
        return self._rho_shift(label, -1)

    def rho_inverse(self, label):
        return self._rho_shift(label, +1)

    def r_label_decompose(self, label):
        """Peel the SU(4) flavour weight off the gauge section.  Returns
        `((h_factors, (0,0,0)), (m1, m2, m3))`."""
        h_factors, w = label
        return (h_factors, (0, 0, 0)), tuple(w)

    def r_label_compose(self, section, r_basis_label):
        h_factors, _ = section
        return (h_factors, tuple(r_basis_label))

    # -- trace ---------------------------------------------------------

    def _trace_residual(self, seed_label, K):
        """ABC stub — see `trace`."""
        return self.trace(seed_label, K)

    def trace(self, label, K: int = 12):
        """Spine-free trace over SU(4), total on every label.

        Identity / Wilson: closed-form Schur F (`tr_W`).  Magnetic gauge
        monomials: the SU(4) character-basis cyclicity+orthonormality bootstrap
        (`su2_nf3_h_trace_bootstrap`, seeded by Tr(1)+Wilson), anchors solved in
        the SU(4) irrep basis and validated against the BPS oracle.

        **Flavour-charged single labels** `(hf, (m1,m2,m3))` with `w≠0`.  A bare
        charged line is **one Cartan section** of a flavour-Weyl (S_4) orbit —
        not an SU(4) class function — so its bare trace is Weyl-COVARIANT,
        centred on `w`.  The flavour Weyl is a symmetry of the canonical basis,
        so the canonical, well-defined physical quantity is the **Weyl-invariant
        section** (recenter to the dominant chamber).  Recentering removes the
        flavour weight, leaving the trace of the neutral gauge seed
        `(hf, (0,0,0))` — which the bootstrap solves directly in the SU(4) irrep
        basis (every q-slice is a verified S_4 class function, e.g.
        `Tr(H_0) = −χ_4̄ q − χ_{(1,0,2)} q³`).  So a charged label routes to its
        neutral-seed trace.

        (The covariant, weight-shifted content of a charged label *inside a
        product* is handled R-linearly by `inner_product` / the bilinear
        `multiply`; this method returns the section — the intrinsic
        class-function content.)
        """
        from su2_nf3_h_trace_bootstrap import trace_label_su4
        h_factors, w = label
        # Charged labels: recenter to the Weyl-invariant section == the neutral
        # gauge-seed trace (same gauge content, flavour weight removed).
        return trace_label_su4((_seed_of(h_factors), (0, 0, 0)), K)

    def inner_product(self, a, b, K: int = 12):
        """`I_{a,b} = Tr(ρ(L_a)·L_b) = δ_{a,b} + O(q)` over SU(4), computed from
        the product decomposition + solved anchors.  Verified δ at q⁰, no
        negative-q, for the magnetic generators (H-tower / Wilson)."""
        from su2_nf3_h_trace_bootstrap import inner_product_su4
        ha, wa = a
        hb, wb = b
        return inner_product_su4((_seed_of(ha), tuple(wa)),
                                 (_seed_of(hb), tuple(wb)), K)


__all__ = ["SU2Nf3ConeKAlgebra"]
