"""`SingleNodeRGKAlgebra`: directional `RGKAlgebra` for single-node BPS-quiver
RG flow.

⚠️  SUPERSEDED (2026-06-10) by
`directional_subquiver_rg.DirectionalSingleNodeRG` /
`DirectionalSubquiverRG` — the modern directional classes: complete
generic API (including the trace this class never implemented and a
non-delegated ρ), closed-form / arranged / unguarded `S_RG` recipes,
F- and S_RG-oracles, and the staged certification harness.  Kept for
existing callers (`a1a2k_induction` et al.); route new work through
`directional_subquiver_rg`.

Per the sharpened `RGKAlgebra` contract: wraps an **IR** `BPSKAlgebra`
(constructed from the UV's spec partitioned into "stuff containing the
dropped node-charge" + "IR spec") and *defines* a UV K-algebra whose
primitives are derived from IR + RG data via `from_ir_image`.

Constructor (same signature shape as `BPSKAlgebra`, with the extra
`gamma_drop` argument identifying the BPS charge of the node to drop):

    SingleNodeRGKAlgebra(pairing, node_charges, spec, gamma_drop)

  * `pairing` -- antisymmetric Z-pairing on the shared lattice Γ.
  * `node_charges` -- UV BPS-quiver nodes (a list of Γ-vectors).
    The node to be dropped must appear here.
  * `spec` -- UV BPS spec, arranged STRICTLY as:
        [stuff_1, ..., stuff_k, ir_1, ..., ir_m]
    where stuff_i are entries decomposing with non-zero coefficient
    on γ = gamma_drop in the `node_charges` basis (positions
    0..k-1), and ir_j are entries with zero coefficient on γ
    (positions k..k+m-1).  Strict: any interleaving raises
    `ValueError`.
  * `gamma_drop` -- the BPS charge (Γ-vector) of the node to drop.
    Must equal `node_charges[j]` for some `j`.

Composition pattern: combine a `SingleNodeRGKAlgebra` with a
`KAlgebraIso` of the wrapped IR `BPSKAlgebra` (to any standalone IR
class, e.g. `Sqed1KAlg ≅ BPSKAlgebra(O→F)`) to produce a
standalone-wrapping RGKAlgebra (e.g., `PentagonSquareRGKAlg`).

Success criterion: `KAlgebraIso(self, BPSKAlgebra(pairing,
node_charges, spec))` -- the constructed directional algebra agrees
with the wrap-UV BPSKAlgebra on multiplicativity / round-trip.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from typing import Sequence

from kalgebra import Element
from laurent_poly import LaurentPoly
from rgkalgebra import RGKAlgebra
from rg_flow import SingleNodeRG, decompose_nonneg_in_node_basis
from bps_kalgebra import BPSKAlgebra

Vec = tuple[int, ...]


class SingleNodeRGKAlgebra(RGKAlgebra):
    """Directional single-node BPS-RG: wraps the IR BPSKAlgebra, defines UV.

    K-algebra primitives `multiply` and (TODO) `trace`-via-Schur-transport
    are inherited from the directional defaults in `RGKAlgebra`.  Label
    structure (`rho`, `identity`, etc.) follows BPSKAlgebra conventions
    on the shared lattice.

    Implementation note: this first cut leverages the existing
    `SingleNodeRG` (wrap-UV shape) internally at construction time to
    extract `RG` and `S_RG`.  The internal UV BPSKAlgebra is an
    extraction tool, not a runtime delegate for K-algebra primitives.
    """

    def __init__(
        self,
        pairing: Sequence[Sequence[int]],
        node_charges: Sequence[Vec],
        spec: Sequence[Vec],
        gamma_drop: Vec,
    ) -> None:
        node_charges = [tuple(g) for g in node_charges]
        spec = [tuple(g) for g in spec]
        gamma_drop = tuple(gamma_drop)
        if gamma_drop not in node_charges:
            raise ValueError(
                f"gamma_drop {gamma_drop} not found in node_charges {node_charges}"
            )
        j_drop = node_charges.index(gamma_drop)

        # Validate spec ordering: stuff (containing γ_drop) MUST precede ir_spec.
        stuff: list[Vec] = []
        ir_spec: list[Vec] = []
        seen_ir = False
        for g in spec:
            decomp = decompose_nonneg_in_node_basis(node_charges, g)
            if decomp is None:
                raise ValueError(
                    f"spec entry {g} doesn't decompose nonneg in node basis"
                )
            if decomp[j_drop] > 0:
                if seen_ir:
                    raise ValueError(
                        f"spec entry {g} (contains γ_drop) appears after an "
                        f"IR-spec entry; required ordering is stuff then ir_spec"
                    )
                stuff.append(g)
            else:
                seen_ir = True
                ir_spec.append(g)

        self._pairing = [list(row) for row in pairing]
        self._node_charges = node_charges
        self._spec = spec
        self._gamma_drop = gamma_drop
        self._j_drop = j_drop
        self._stuff_spec = stuff
        self._ir_spec = ir_spec

        # Build IR BPSKAlgebra (this IS `auxiliary()`).
        ir_node_charges = (
            node_charges[:j_drop] + node_charges[j_drop + 1:]
        )
        self._aux = BPSKAlgebra(
            pairing=pairing,
            node_charges=ir_node_charges,
            spec=ir_spec,
            verify="off",
        )

        # Build UV BPSKAlgebra internally (extraction tool only).
        self._uv_ext = BPSKAlgebra(
            pairing=pairing,
            node_charges=node_charges,
            spec=spec,
            verify="off",
        )

        # Build SingleNodeRG internally (extraction tool only) to get
        # RG and rg_generator.
        self._rg_ext = SingleNodeRG(self._uv_ext, node_index=j_drop)

        # Cache RG(L_a) -- the directional multiply hits this repeatedly.
        self._rg_cache: dict[Vec, Element] = {}

    # ----- KAlgebra contract: label structure -----------------------------

    def coefficient_ring(self):
        return self._uv_ext.coefficient_ring()

    def identity(self):
        return self._uv_ext.identity()

    def rho(self, a):
        return self._uv_ext.rho(a)

    def rho_inverse(self, a):
        return self._uv_ext.rho_inverse(a)

    def _label_section_decompose(self, label):
        return self._uv_ext._label_section_decompose(label)

    def trace(self, a, K=20):
        """Not yet implemented; will derive from IR via Schur transport."""
        raise NotImplementedError(
            "SingleNodeRGKAlgebra.trace via Schur transport: TODO."
        )

    # ----- RGKAlgebra contract --------------------------------------------

    def auxiliary(self):
        return self._aux

    def RG(self, a) -> Element:
        a = tuple(a)
        if a not in self._rg_cache:
            self._rg_cache[a] = self._rg_ext.RG(a)
        return self._rg_cache[a]

    def rg_generator(self, cutoff: int) -> dict:
        return self._rg_ext.rg_generator(cutoff)

    def from_ir_image(self, x_ir: Element) -> Element:
        """Recognise an IR `Element` as a UV `Element` via tropical-peeling.

        Strategy: at each step pick a "leading" IR canonical-basis term
        in the residual (max under lex on the lattice tuple).  The UV
        label whose RG-image leading term is this IR label is the same
        charge γ (shared lattice).  Compute `RG(L^UV_γ)`, match the
        coefficient, subtract, repeat.

        For BPS single-node RG the leading-term bijection on the shared
        lattice holds by inspection.  A general directional implementation
        without the leading-term bijection would use linear-algebra
        decomposition over Z[q±]; deferred.
        """
        result: dict[Vec, LaurentPoly] = {}
        residual = {k: v for k, v in x_ir.terms.items() if not v.is_zero()}
        max_iters = 5000
        iters = 0
        while residual and iters < max_iters:
            iters += 1
            ir_label = max(residual.keys())
            uv_label = ir_label
            rg_uv = self.RG(uv_label)
            if ir_label not in rg_uv.terms:
                raise RuntimeError(
                    f"from_ir_image: ir_label {ir_label} not in "
                    f"RG({uv_label}); leading-term bijection failure."
                )
            rg_coef = rg_uv.terms[ir_label]
            res_coef = residual[ir_label]
            items = list(rg_coef._coeffs.items())
            if len(items) != 1 or items[0][1] not in (1, -1):
                raise NotImplementedError(
                    f"from_ir_image: rg_coef {rg_coef} is not ±q^k monomial; "
                    "general polynomial inversion not implemented."
                )
            k, c = items[0]
            alpha = res_coef * LaurentPoly({-k: c})
            result[uv_label] = (
                result.get(uv_label, LaurentPoly.zero()) + alpha
            )
            for lbl, lc in rg_uv.terms.items():
                new_c = residual.get(lbl, LaurentPoly.zero()) - alpha * lc
                if new_c.is_zero():
                    residual.pop(lbl, None)
                else:
                    residual[lbl] = new_c
        if iters >= max_iters:
            raise RuntimeError(
                f"from_ir_image: did not converge after {max_iters} iters; "
                f"residual={residual}"
            )
        return Element({k: v for k, v in result.items() if not v.is_zero()})

    # ----- introspection --------------------------------------------------

    @property
    def stuff_spec(self) -> list[Vec]:
        return list(self._stuff_spec)

    @property
    def ir_spec(self) -> list[Vec]:
        return list(self._ir_spec)

    @property
    def gamma_drop(self) -> Vec:
        return self._gamma_drop

    @property
    def j_drop(self) -> int:
        return self._j_drop
