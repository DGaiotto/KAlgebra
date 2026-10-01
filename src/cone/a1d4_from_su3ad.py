"""A₁D₄ K-algebra as a flavour restriction of the SU(3)-enhanced
[A₁, D₄] algebra:  `a1d4 = SU3ADKAlg.base_change(su3_to_su2u1_hom())`.

Why this construction (regen note / Plan-18 D2, user-approved): the
cluster-BFS pipeline cannot build a1d4 — the SU(2)×U(1) section does
not carry the D₄-triality ℤ₃ that closes the cluster graph at 8
mult-gens, so `finite_a1d4_kalg.py` froze a TRUNCATION (3 mg's).  The
full algebra is the SU(3)-enhanced `SU3ADKAlg` (8 mg's) with the
flavour symmetry *restricted* along the block embedding
`SU(2)×U(1) ⊂ SU(3)` (fundamental branching `3 → 2₊₁ ⊕ 1₋₂`) — a
`KAlgebra.base_change` in the contract's sense (functorial
coefficient-ring change = physical flavour restriction).

Upstream status (2026-06-13): the source `SU3ADKAlg` is now REPAIRED —
`verify_rho_is_automorphism` holds on all generator pairs (the static
relation tables were missing the per-ρ-orbit-position χ-parity ⋆; see
`su3_ad_kalg._chi_parity`), the multiply is certified product-for-
product against `SU3BPSKAlgebra`, and `trace(1)` carries the χ_(1,1)
adjoint at q².  Because the restriction is exactly functorial (the
image's structure is the φ-image of the source's), this module's
algebra inherited the repair with no change here.
"""
from __future__ import annotations

from su3_ad_kalg import SU3ADKAlg
from zplus_ring import su3_to_su2u1_hom


def a1d4_kalgebra():
    """The A₁D₄ K-algebra over `SU2xU1ZPlusRing`, built by flavour
    restriction of `SU3ADKAlg` along `su3_to_su2u1_hom`."""
    return SU3ADKAlg().base_change(su3_to_su2u1_hom())


__all__ = ["a1d4_kalgebra"]
