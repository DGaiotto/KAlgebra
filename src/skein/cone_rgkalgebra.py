"""
cone_rgkalgebra.py
==================

`ConeRGKAlgebra(ConeKAlgebra, RGKAlgebra)` — fourth-and-a-half
presentation tier of the `KAlgebra` hierarchy.

`ConeRGKAlgebra` multiply-inherits both `ConeKAlgebra` (cone-data
presentation: cone structure + canonical-basis combinatorics) and
`RGKAlgebra` (RG flow to an auxiliary IR algebra).  The result is a
K-algebra with:

  * cone-data declarative structure (mult-gens, q-commute graph,
    cocycle, cones) on the UV side, providing
    `cone_data().derived_multiply` for arbitrary cone-monomial
    products and the Layer-1 tagged-cyclicity trace reducer;
  * RG-driven analytic primitives (`auxiliary`, `RG`, `rg_generator`,
    `from_ir_image`) providing both a multiply-via-IR-image route
    AND the seed-trace evaluation (Layer-2) via
    `auxiliary().trace(RG(seed), K)`.

The two interfaces are merged by Python's MRO so that:
  * `multiply` defaults to `ConeKAlgebra.multiply` (= cone-data
    `derived_multiply`).  Subclasses MAY override with the
    `RGKAlgebra` directional default (`from_ir_image(aux.multiply(
    RG(a), RG(b)))`) if the cone-data `cross_product` table isn't
    available or RG-mediation is preferred — see
    `PentagonSquareKAlg`.
  * `trace` defaults to `ConeKAlgebra.trace` (= cone-data Layer 1 +
    `_trace_residual`).  `ConeRGKAlgebra` provides a default
    `_trace_residual` that delegates to the RG side; subclasses
    can override.

Use cases
---------
* `PentagonSquareKAlg` (= `PentagonSquareRGKAlg`): Pentagon UV with
  Sqed1 IR, multiply via RG image, trace via RG-side Schur
  transport.
* `U1HexagonPentagonKAlg`: U1Hexagon UV with Pentagon IR (cluster
  RG flow folding the torus direction).
* BPS-style algebras with a quantum-torus IR.

Naming convention
-----------------
`ConeRGKAlgebra` subclasses end in `KAlg` (NOT `RGKAlg`): they are
genuine K-algebras presented via cone-data + RG-flow.  This
distinguishes them from pure `RGKAlgebra` subclasses (which name
typically ends in `RG`).
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cone_kalgebra import ConeKAlgebra
from rgkalgebra import RGKAlgebra
from kalgebra import Label
from zplus_ring import RPowerSeries, RLaurent, RingHom


__all__ = ["ConeRGKAlgebra"]


class ConeRGKAlgebra(ConeKAlgebra, RGKAlgebra):
    """K-algebra presented via cone-data + RG flow.

    Multiply-inherits `ConeKAlgebra` (cone-data combinatorics) and
    `RGKAlgebra` (RG flow to auxiliary IR algebra).  Concrete
    subclasses implement:

      * From `ConeKAlgebra`:
          `cone_data()` -- a `ConeData` instance.
      * From `RGKAlgebra`:
          `auxiliary()`, `RG(a)`, `rg_generator(cutoff)`,
          `from_ir_image(x_ir)`.
      * From `KAlgebra` (still abstract through both parents):
          `coefficient_ring()`, `identity()`, `rho`, `rho_inverse`,
          `_label_section_decompose`.

    `multiply` and `trace` are inherited from `ConeKAlgebra` by
    default (MRO).  `_trace_residual` has a default RG-side
    implementation (see below) which subclasses may override or
    short-circuit.
    """

    # ----- default Layer-2 trace via RG -------------------------------

    def _trace_residual(self, seed_label: Label, K: int) -> RPowerSeries:
        """Default `_trace_residual` for `ConeRGKAlgebra` subclasses:
        evaluate the seed trace in the IR auxiliary algebra via the
        RG image.

            Tr_UV(L_seed)  =  Tr_IR(RG(L_seed))

        valid when the RG flow descends to a trace-preserving map
        between UV and IR canonical-basis traces (the cases this
        contract is built for satisfy this; see e.g.
        `PentagonSquareKAlg.trace` for the Schur-transport
        derivation).

        Subclasses with closed-form Layer-2 (e.g., a direct character
        formula not going through the IR) should override.
        """
        rg_seed = self.RG(seed_label)
        # Linear extension of `auxiliary().trace` to an `Element`:
        # uses the inherited `trace_element` from `KAlgebra`.
        return self.auxiliary().trace_element(rg_seed, K)
