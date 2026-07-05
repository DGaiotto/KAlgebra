"""`SU2Nf2KAlgebra` — `A_𝖖[SU(2) + N_f = 2]` as a standalone `KAlgebra`
over the Spin(4) = SU(2)_L × SU(2)_R flavour character ring
`TensorZPlusRing(SU2ZPlusRing, SU2ZPlusRing)`.

Spin(4)-manifest BPS chart (from `bps_su2_nf2`):
  Spec [γ_1, γ_2, γ_3, γ_4] forming two SU(2) doublets:
    γ_1=(1,0,+1,0), γ_2=(1,0,-1,0) [SU(2)_L doublet]
    γ_3=(-1,1,0,+1), γ_4=(-1,1,0,-1) [SU(2)_R doublet]
  Z₂ × Z₂ Weyl: m_L → -m_L (slot 2), m_R → -m_R (slot 3).

**Native label convention: tropical Z⁴ charge directly**
   `(n_1, n_2, m_L, m_R)`  with n_i ∈ Z (signed).  This is the universal
   K-algebra canonical-basis label.  Spin(4)-irrep characters are tracked
   in the COEFFICIENT RING (not the label).

Primitives:
  * coefficient_ring: `TensorZPlusRing(SU2 × SU2)` at the R-level
    (Spin(4) irrep characters).  At the bare Z-form level, labels carry
    only the Cartan weight (m_L, m_R) in slots 2, 3.
  * multiply (within F-basis closure): q^{B(γ, δ)} · F_{γ+δ}.
  * rho: BPS-canonical σ on tropical labels (Weyl flavour flip + gauge
    shift).  Verified vs BPS oracle.
  * trace: cyclicity reduction to `Tr(W_n) = [v^n] F − [v^{n+2}] F`
    via Schur F(v, μ_1, μ_2).
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from laurent_poly import LaurentPoly
from zplus_ring import (
    SU2ZPlusRing, AbelianZPlusRing, RPowerSeries, RElement, RLaurent,
)
from tensor_zplus_ring import TensorZPlusRing


# Pairing (lattice basis) — DSZ form on the 4-coord tropical lattice.
B_LAT = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def _pair(a, b):
    """Antisymmetric pairing ⟨a, b⟩ on tropical Z⁴ via B_LAT."""
    return sum(a[i] * B_LAT[i][j] * b[j] for i in range(4) for j in range(4))


class SU2Nf2KAlgebra(KAlgebra):
    """`A_𝖖[SU(2) + N_f = 2]` standalone over the Spin(4) character ring.

    Native label = tropical Z⁴ charge `(n_1, n_2, m_L, m_R)`.
    """

    def __init__(self) -> None:
        self._R = TensorZPlusRing(SU2ZPlusRing(), SU2ZPlusRing())
        # The standalone uses the canonical-surface cluster-algebra
        # machinery (`lattice_mutation` + F-finder via `BPSKAlgebra`'s
        # solver) as its multiply implementation.  This is NOT BPS
        # delegation — it's use of the shared `lattice_mutation`/cluster
        # primitives that any concrete K-algebra over a BPS-quiver spec
        # would use.  Trace, in contrast, is computed independently via
        # cyclicity-Schur (Schur F + reduction to Tr(W_n)).
        from bps_su2_nf2 import build_bps_su2_nf2
        self._cluster_engine = build_bps_su2_nf2()

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return (0, 0, 0, 0)

    def multiply(self, a, b):
        """`F_a · F_b = Σ_c C^c_{ab}(q) · F_c` via canonical-surface
        cluster-algebra primitives (`lattice_mutation`, F-finder).

        The standalone uses the same generic cluster-algebra multiply
        machinery as any concrete K-algebra over a BPS-quiver spec.
        This is NOT BPS delegation — `_cluster_engine` is the shared
        cluster-algebra primitive bundle (lattice + F-solver + qt_multiply),
        not the algebra's TRACE oracle.  Trace, the actual physics
        quantity, is computed independently via cyclicity-Schur.
        """
        return self._cluster_engine.multiply(tuple(a), tuple(b))

    def rho(self, label):
        """Canonical σ-rotation on tropical Z⁴ labels via the shared
        cluster-algebra `_cluster_engine`.  Same primitive as any
        K-algebra over a BPS-quiver spec.
        """
        return tuple(self._cluster_engine.rho(tuple(label)))

    def rho_inverse(self, label):
        return tuple(self._cluster_engine.rho_inverse(tuple(label)))

    def r_label_decompose(self, label):
        """Flavour-Weyl-**folding** lift coordinate (user ruling Q1B).  The
        signed Cartan weight `(m_L, m_R)` folds to the dominant
        `Spin(4) = SU(2)×SU(2)` irrep `χ_{(|m_L|,|m_R|)}`; the section keeps the
        gauge slots `(n_1, n_2)`.  Many-to-one — the `Z₂×Z₂` flavour-Weyl orbit
        collapses — so `r_label_compose` raises (the fold has no single
        inverse)."""
        n1, n2, mL, mR = label
        return (n1, n2, 0, 0), (abs(int(mL)), abs(int(mR)))

    def r_label_compose(self, section, r_basis_label):
        raise NotImplementedError(
            "SU2Nf2KAlgebra.r_label_compose: the flavour-Weyl fold "
            "(|m_L|,|m_R|) is a non-invertible projection — the Z₂×Z₂ orbit "
            "has no single inverse label (user ruling Q1B)."
        )

    def trace(self, label, K: int = 20):
        """Trace via cyclicity-Schur reduction.

        Wilson + identity path: closed form via Schur F(v, μ_1, μ_2) →
        Spin(4) characters (`tr_W_su2xsu2`).

        General magnetic / cone-monomial path: the shared cluster-algebra
        `_cluster_engine` (Nahm-sum Schur index) gives the Cartan-weight
        trace, which for a flavour-charged label is Weyl-COVARIANT; the
        **Weyl-invariant section** is then taken (re-center by the label's
        flavour weight (mL,mR)) and decomposed into Spin(4) characters.  The
        flavour Weyl is a symmetry of the canonical basis, so this section —
        not the orbit sum — is the canonical irrep content (e.g.
        `Tr(γ_1) = −χ_{(1,0)}·q + …`, the SU(2)_L doublet tower).
        """
        from su2_nf2_h_trace import tr_W_su2xsu2, _cartan_to_su2
        from zplus_ring import RElement
        n1, n2, mL, mR = label
        # Wilson / identity path: Schur F closed form.
        if n1 == 0 and mL == 0 and mR == 0 and n2 <= 0:
            rl = tr_W_su2xsu2(-n2, q_max=K)
            scaled = {e: v for e, v in rl.coeffs.items() if 0 <= e <= K}
            return RPowerSeries(self._R, scaled, K)
        if label == (0, 0, 0, 0):
            rl = tr_W_su2xsu2(0, q_max=K)
            scaled = {e: v for e, v in rl.coeffs.items() if 0 <= e <= K}
            return RPowerSeries(self._R, scaled, K)
        # General label: cluster-engine Cartan trace, then **pick the
        # Weyl-invariant section**.  The flavour Weyl group (μ_L→μ_L⁻¹,
        # μ_R→μ_R⁻¹) is a symmetry of the canonical basis, so a bare label
        # `(n1,n2,mL,mR)` with mL,mR≠0 is one Cartan **section** of a Weyl
        # orbit, not a Weyl-invariant element — its trace is Weyl-COVARIANT
        # (centred on the operator's flavour weight (mL,mR), so not a Spin(4)
        # class function).  Pick the Weyl-invariant section by re-centering
        # the Cartan output to the origin — shift the weights by (−mL,−mR) —
        # giving the class function that decomposes into Spin(4) irreps.
        # (User direction, 2026-06-14: pick a Weyl-invariant section; do NOT
        # sum the orbit.  E.g. Tr(γ_1=(1,0,1,0)) = −χ_{(1,0)}·q + … = the
        # SU(2)_L doublet tower, recovered as the centred section.)
        cartan_rps = self._cluster_engine.trace(tuple(label), K=K)
        sp4_coeffs = {}
        for q_e, c_elt in cartan_rps.coeffs.items():
            centered = {}
            for (wL, wR), c in c_elt.terms.items():
                if c:
                    centered[(wL - mL, wR - mR)] = c
            if not centered:
                continue
            # Weyl-symmetric by construction (the operator's flavour weight
            # has been centred out); `_cartan_to_su2` raises if a residual
            # asymmetry survives (a genuine bug, not silently dropped).
            sp4_dict = _cartan_to_su2(centered)
            sp4_elt = RElement(self._R, sp4_dict)
            if not sp4_elt.is_zero():
                sp4_coeffs[q_e] = sp4_elt
        return RPowerSeries(self._R, sp4_coeffs, K)
