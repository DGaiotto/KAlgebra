"""`RGKAlgebraObject`s for the SU2A1D3 single-node RG flows (the design record object layer
applied to flows).

Each object holds **one** node-deletion flow of SU(2)-gauged [A_1,D_3] in two live
descriptions — the wrapped BPS IR and the certified standalone-IR swap — bridged by
the `RGKAlgebraIso` flow witness (UV labels shared = identity; the aux leg is the
verified IR `KAlgebraIso`).

* :func:`su2a1d3_matter_drop_object` — **drop the matter node (2) on the det-1
  line lattice `su2a1d3_det1()`: IR = pure SU(2) × SQED₁ (FULL, monopole on-cone).**
  The IR auxiliary factors as a clean tensor `pure SU(2) ⊗ Sqed1KAlg`: pure SU(2) in
  the first `Z²` (`(a,b)`, Kronecker `{(1,0),(-1,2)}`), SQED₁ in the second `Z²`
  (`(c,d)`: electric `(0,0,0,1)`, magnetic monopole `(0,0,1,0)`, pairing 1); the
  factors q-commute.  The IR `KAlgebraIso` φ — split `(a,b,c,d) ↔ ((a,b),(c,d))` —
  passes the **whole battery incl. trace**; the flow witness `rg_intertwine` and the
  IsoComposed swap's `verify_rg_unital`/`multiplicative` hold **including the monopole
  sector** `(0,0,1,0)`.

  Γ = `Z²⊕Z²` is fixed and mutation-preserved; the node charges span an index-2
  sublattice (the gauge Wilson/monopole are half-elements, exactly as for pure
  SU(2)'s standard `(1,0),(-1,2)`).  The RG **grading** reads the degree as the
  dropped-direction coefficient (the unique `c`-carrier ⇒ always integer), Γ-aware
  via `directional_subquiver_rg._rational_node_basis_decompose` — so the monopole
  grades correctly.

* :func:`su2a1d3_tail_drop_object` — **drop the tail/U(1) node (3): IR = U(2)
  N_f=1** (coarse build), swapped for the certified `flavour_free_un_nf_bps(2,1)`
  (PR).

Run `PYTHONPATH=$(ls -d src/* | paste -sd:) python3 src/abe/su2a1d3_rg_object.py` for the smoke test.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element
from laurent_poly import LaurentPoly
from rgkalgebra_object import RGKAlgebraObject, RGKAlgebraIso
from iso_composed_rgkalgebra import IsoComposedRGKAlgebra

from su2a1d3_gauged import (
    matter_drop_det1_iso_to_pure_su2_sqed1,
    matter_drop_det1_iso_to_abe_su2_sqed1,
    tail_drop_iso_to_u2nf1,
)

__all__ = [
    "su2a1d3_matter_drop_object",
    "su2a1d3_matter_drop_abe_object",
    "su2a1d3_tail_drop_object",
]

_ONE = LaurentPoly.one()


def _id_label(label):
    return Element({tuple(label): _ONE})


def _drop_flow_object(flow, phi, *, name, ir_key, ir_caps):
    """Assemble the RGKAlgebraObject for one drop flow: `flow` (wrapped BPS IR) +
    its IsoComposed standalone-IR swap, bridged by the flow witness whose aux leg
    is `phi` (the verified IR `KAlgebraIso`, aux → standalone IR)."""
    swapped = IsoComposedRGKAlgebra(flow, phi)
    witness = RGKAlgebraIso(flow, swapped, _id_label, _id_label, phi,
                            name=f"{name}[bps-ir→{ir_key}]")
    obj = RGKAlgebraObject(name)
    obj.add_realization("bps-ir", flow, {"rg", "bps-ir"})
    obj.add_realization(ir_key, swapped, ir_caps)
    obj.add_iso("bps-ir", ir_key, witness)
    return obj


def su2a1d3_matter_drop_object() -> RGKAlgebraObject:
    """Matter-drop flow on the det-1 line lattice (drop node 2): IR = **pure SU(2)
    × SQED₁** (full, monopole on-cone), realized at `pure SU(2) ⊗ Sqed1KAlg`."""
    flow, phi, _target = matter_drop_det1_iso_to_pure_su2_sqed1()
    return _drop_flow_object(
        flow, phi,
        name="SU2A1D3 matter-drop → pure SU(2) × SQED₁ (det-1, full)",
        ir_key="pure-su2-x-sqed1",
        ir_caps={"rg", "standalone-ir", "factored", "pure-su2", "sqed1"})


def su2a1d3_matter_drop_abe_object() -> RGKAlgebraObject:
    """Matter-drop flow with the pure-SU(2) IR factor **ABELIANIZED**: IR aux =
    `AbelianizedSU2KAlg ⊗ Sqed1KAlg` (`SU2A1D3MatterDropAbeRG`).  Same RG/multiply
    content as `su2a1d3_matter_drop_object` (the swap iso is multiplicative /
    ρ-equivariant, monopole graded); the pure-SU(2) leg is the abelianized quantum
    torus instead of the BPS Kronecker.  Trace is NOT equivariant (the abelianized
    trace is the U(2) Schur index — decoupled-photon factor), as documented for
    `AbelianizedSU2KAlg`."""
    flow, phi_abe, _t = matter_drop_det1_iso_to_abe_su2_sqed1()
    return _drop_flow_object(
        flow, phi_abe,
        name="SU2A1D3 matter-drop → (abelianized pure SU(2)) × SQED₁ (det-1)",
        ir_key="abe-su2-x-sqed1",
        ir_caps={"rg", "standalone-ir", "factored", "abelianized", "sqed1"})


def su2a1d3_tail_drop_object() -> RGKAlgebraObject:
    """Tail/U(1)-drop flow (drop node 3): IR = U(2) N_f=1, swapped for the certified
    `flavour_free_un_nf_bps(2,1)` standalone (PR)."""
    flow, phi, _u2 = tail_drop_iso_to_u2nf1()
    return _drop_flow_object(
        flow, phi,
        name="SU2A1D3 tail-drop → U(2) N_f=1",
        ir_key="u2-nf1",
        ir_caps={"rg", "standalone-ir", "u2-nf1"})


if __name__ == "__main__":
    print("== matter-drop → pure SU(2) × SQED₁ (det-1, full) ==")
    O = su2a1d3_matter_drop_object()
    print(O, "\n  keys:", O.keys())
    w = O.iso("bps-ir", "pure-su2-x-sqed1")
    uv = [(1, 0, 0, 0), (-1, 2, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1)]
    print("  verify_rg_intertwine (incl. monopole):", w.verify_rg_intertwine(uv))
    swap = O.realization("pure-su2-x-sqed1")
    print("  swap rg_unital:", swap.verify_rg_unital())
    print("  swap rg_multiplicative (monopole·electric):",
          swap.verify_rg_multiplicative((0, 0, 1, 0), (0, 0, 0, 1)))

    print("\n== matter-drop → (ABELIANIZED pure SU(2)) × SQED₁ (det-1) ==")
    Oa = su2a1d3_matter_drop_abe_object()
    print(Oa, "\n  keys:", Oa.keys())
    aux = Oa.realization("abe-su2-x-sqed1").auxiliary()
    print("  IR aux factor_A:", type(aux.factor_A).__name__)
    wa = Oa.iso("bps-ir", "abe-su2-x-sqed1")
    print("  verify_rg_intertwine (incl. monopole):", wa.verify_rg_intertwine(uv))

    print("\n== tail-drop → U(2) N_f=1 ==")
    T = su2a1d3_tail_drop_object()
    print(T, "\n  keys:", T.keys())
    print("  verify_rg_intertwine:", T.iso("bps-ir", "u2-nf1").verify_rg_intertwine(
        [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)]))
