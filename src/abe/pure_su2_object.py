"""`KAlgebraObject` for **pure SU(2) gauge theory**: the abstract `A_𝖖[SU(2)]` holding its certified
presentations under one roof.

Realizations (all pre-existing — assembled, not re-derived):

* ``'abe'``  — `AbelianizedSU2KAlg`: the **AbeKAlgebra-side** light
  wrapper over the pure-U(2) abelianized algebra `PureGAbeKAlgebra(u_n(2))` (over
  the retired `PureUNKAlgebra(2)` keystone until 2026-09-19): SU(2)
  lines `L_{m,e}` on the trace-zero sublattice `m·(1,−1)`, structure
  delegated and read back through the det-projection (f-presentation
  native; no DOp).
* ``'cone'`` — `PureSU2KAlg`: the **ConeKAlgebra-side** H-tower
  realization (rank-3 cone data `{H_{2k}, H_{2k+1}, H_{2k+2}}` per
  cone, Wilson cone `w₁`, axiom-derived literal-word multiply and
  cyclicity-bridge trace).
* ``'bps'``  — `BPSKAlgebra` on the **Kronecker quiver with node
  charges `(1,0)` and `(−1,2)`** (= `pure_ade_kalgebra([("A",1)])`;
  `⟨γ₁,γ₂⟩ = 2`).
* ``'skein'`` — `SkeinAnnulusKAlg`: the **stated-skein annulus**
  realization (one marked point per boundary circle — the Kronecker
  quiver's cluster surface; the formerly documented open slot, landed
  2026-07-08 on the certified `CutReglue` bordered chart of the
  once-punctured-torus cut, not the pinned-polygon engine — the
  annulus chart's dual graph has a 2-cycle).  Labels = the bps lower
  tropical charges verbatim; `multiply` is computed GENUINELY in the
  extended bordered chart and reduced by the flavour projection `P`
  with the standing verbatim-assert guard (δ ≡ 0); ρ and trace are
  NATIVE (ρ = P∘ρ_B∘Γ; trace = ε∘Tr_B∘Γ, ε the flavour
  augmentation).  The measured χ₂ surprise: the once-wrapping core
  multicurve is the ADJOINT Wilson `L_{0,2}`; χ₁ enters at the
  boundary-unit canonical (its geometric stated realization stays
  conjectural — `skein_sphere/skein_annulus_kalg.py`).

Witnesses (each the full `KAlgebraIso` battery):

* ``abe ↔ bps`` — the pre-existing certified `abelianized_su2_bps_iso`
  (label maps `gamma_of`/`me_of`; the dyon node `γ₂ = (−1,2)` is
  `L_{1,2}`).
* ``cone ↔ abe`` — the documented H-tower dictionary
  `_native_to_psu2` / `_psu2_to_native`: an H-monomial
  `∏ H_{nᵢ}^{aᵢ}` is the line `(m, e) = (Σaᵢ, Σ nᵢaᵢ)`; Wilson
  natives `('W', e)` are `(0, e)`; inverse by the max-diagonal cone
  algorithm.  (Identity-coefficient label bijection; the battery is
  the certificate that the cone's axiom-derived multiply/ρ agree with
  the keystone-delegated ones.)
* ``cone ↔ bps`` — the composite through the abe hub, stored
  explicitly so the groupoid triangle is full and `verify_coherence`
  is non-vacuous.
* ``skein ↔ bps`` — `SkeinAnnulusKAlg.build_iso()`: identity label
  maps on the shared lower-tropical labels (the skein leg is
  CONSTRUCTED on the registered bps instance, so the iso endpoints
  are the registered realizations); the battery + the 144-product
  grid live in `skein_sphere/the suite in the source repository.

Trace caveat (inherited, documented): `AbelianizedSU2KAlg.trace` is
the U(2) Schur index (decoupled-photon factor) — the abe↔bps battery
certifies the photon-independent structure, exactly as in
`abelianized_su2_bps_iso`.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly

from abelianized_su2_kalgebra import AbelianizedSU2KAlg, su2_fold
from abelianized_su2_bps_iso import abelianized_su2_bps_iso
from pure_su2_h_cone_data import (
    PureSU2KAlg, _native_to_psu2, _psu2_to_native)
from skein_annulus_kalg import SkeinAnnulusKAlg


__all__ = ["pure_su2_object"]

_ONE = LaurentPoly.one()


def pure_su2_object(K: int = 12) -> KAlgebraObject:
    """The abstract pure-SU(2) algebra as a `KAlgebraObject` (abe /
    cone / bps / skein legs wired)."""
    obj = KAlgebraObject("A_q[pure SU(2)]")

    # --- abe: the AbeKAlgebra-side wrapper over the keystone ---------
    iso_ab = abelianized_su2_bps_iso(K=K)
    abe = iso_ab.source
    bps = iso_ab.target
    obj.add_realization("abe", abe, {"chart", "multiply-fast",
                                     "f-presentation"})

    # --- bps: Kronecker (1,0)/(−1,2) ---------------------------------
    assert list(getattr(bps, "node_charges")) == [(1, 0), (-1, 2)]
    obj.add_realization("bps", bps, {"chart", "rg", "trace-exact"})
    obj.add_iso("abe", "bps", iso_ab)

    # --- cone: the H-tower cone-data realization ----------------------
    cone = PureSU2KAlg()
    obj.add_realization("cone", cone, {"multiply-fast", "closed-form"})

    def _cone_fwd(native):
        return Element({su2_fold(*_native_to_psu2(native)): _ONE})

    def _cone_inv(label):
        m, e = label
        return Element({_psu2_to_native(m, e): _ONE})

    obj.add_iso("cone", "abe", KAlgebraIso(
        cone, abe,
        forward_label_map=_cone_fwd,
        inverse_label_map=_cone_inv,
        name="pure-su2[cone→abe]  (H-monomial (m,e) = (Σa, Σna) dictionary)"))

    # close the triangle (composite through the abe hub), so the
    # coherence certificate is non-vacuous
    obj.add_iso("cone", "bps", obj.iso("cone", "bps"))

    # --- skein: the stated-annulus bordered-chart realization ---------
    # constructed ON the registered bps instance, so the iso endpoints
    # ARE the registered realizations (instance identity)
    sk = SkeinAnnulusKAlg(bps=bps)
    obj.add_realization("skein", sk, {"chart", "skein"})
    obj.add_iso("skein", "bps", sk.build_iso())

    return obj


if __name__ == "__main__":
    obj = pure_su2_object()
    print(obj)
    print("keys:", obj.keys())
    H0, H1, W1 = ((0, 1),), ((1, 1),), ((("W", 1), 1),)
    for lab in [H0, H1, W1, ((0, 2),), ((0, 1), (1, 1))]:
        print(f"  cone {lab} → abe {obj.transport(lab, 'cone', 'abe').terms}"
              f" → bps {obj.transport(lab, 'cone', 'bps').terms}")
