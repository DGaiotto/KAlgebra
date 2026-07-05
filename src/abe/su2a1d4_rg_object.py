"""Object-layer (Plan 25) views of the SU2A1D4 (det-1) single-node RG flows.

* :func:`su2a1d4_tail_drop_object` — the **forgetful-RG ladder SU2A1D4 → SU2A1D3**.
  Dropping the tail-end node (the only flavour-charged node) gives an IR whose
  canonical basis is flavour-neutral and equals `su2a1d3_det1`; the swap forgets the
  residual U(1) flavour.  An `RGKAlgebraObject` (wrapped BPS IR + the IsoComposed
  swap to `su2a1d3_det1`), flow-witness `rg_intertwine` verified.

The matter-drop flow (→ pure SU(2) × (SQED₁ + flavour)) is exposed as the verified
`SingleNodeRG` `su2a1d3_gauged.su2a1d4_det1_matter_drop_flow`; its IR factorisation
is checked in `tests/test_su2a1d4_rg.py`.

Run `PYTHONPATH=. python implementations/su2a1d4_rg_object.py` for the smoke test.
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

from su2a1d3_gauged import su2a1d4_tail_drop_iso_to_su2a1d3

__all__ = ["su2a1d4_tail_drop_object"]

_ONE = LaurentPoly.one()


def _id_label(label):
    return Element({tuple(label): _ONE})


def su2a1d4_tail_drop_object() -> RGKAlgebraObject:
    """The forgetful-RG ladder SU2A1D4 → SU2A1D3 as an `RGKAlgebraObject`:
    `bps-ir` (the det-1 tail-drop flow) + `su2a1d3` (the IsoComposed swap to
    `su2a1d3_det1`), bridged by the flow witness whose aux leg is the structural
    forgetful iso `phi`."""
    flow, phi, _A3 = su2a1d4_tail_drop_iso_to_su2a1d3()
    swapped = IsoComposedRGKAlgebra(flow, phi)
    witness = RGKAlgebraIso(flow, swapped, _id_label, _id_label, phi,
                            name="SU2A1D4 tail-drop[bps-ir→su2a1d3]")
    obj = RGKAlgebraObject("SU2A1D4 tail-drop → SU2A1D3 (forgetful RG)")
    obj.add_realization("bps-ir", flow, {"rg", "bps-ir"})
    obj.add_realization("su2a1d3", swapped,
                        {"rg", "standalone-ir", "su2a1d3", "forgetful"})
    obj.add_iso("bps-ir", "su2a1d3", witness)
    return obj


if __name__ == "__main__":
    O = su2a1d4_tail_drop_object()
    print(O, "\n  keys:", O.keys())
    w = O.iso("bps-ir", "su2a1d3")
    uv = [(1, 0, 0, 0, 0), (-1, 2, 0, 0, 0), (0, -1, 1, 0, 0), (0, 0, 0, 1, 0)]
    print("  verify_rg_intertwine:", w.verify_rg_intertwine(uv))
    swap = O.realization("su2a1d3")
    print("  swap aux:", type(swap.auxiliary()).__name__,
          "ring:", swap.auxiliary().coefficient_ring())
    print("  swap rg_unital:", swap.verify_rg_unital())
