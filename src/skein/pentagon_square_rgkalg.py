"""`PentagonSquareKAlg`: `PentagonKAlg` with `multiply` and `trace`
replaced by RG-mediated versions through `Sqed1KAlg`.

Inheritance: `PentagonSquareKAlg(PentagonKAlg, ConeRGKAlgebra)`.
Promoted from `PentagonSquareRGKAlg(PentagonKAlg, RGKAlgebra)` —
`PentagonSquareRGKAlg` is kept as a backward-compat alias at the
bottom of the module.

Inherited from `PentagonKAlg` (no duplication):
  * Label structure `(i, a, b)` and canonicalisation.
  * `coefficient_ring()`, `identity()`, `rho`, `rho_inverse`,
    `_label_section_decompose`, `cone_data()`.

Added (RGKAlgebra contract):
  * `auxiliary()` -> `Sqed1KAlg()`
  * `RG(a)` -> multiplicative extension of hard-coded `RG_GENERATORS`.
  * `rg_generator(cutoff)` -> `E_q(u_-)` Habiro tower.
  * `from_ir_image(x_ir)` -> peel Pentagon labels from Sqed1 elements.

Overridden:
  * `multiply` -> RGKAlgebra's directional default
    (`from_ir_image(aux.multiply(RG(a), RG(b)))`).
  * `trace(a, K)` -> Schur transport in Sqed1:
        `trace_UV(a) = trace_IR(ρ_IR(S_RG) · RG(a)·S_RG)`
                     = `inner_product_IR(S_RG, RG(a)·S_RG)`.

The **only** hand-coded family-specific data is `RG_GENERATORS`
(5 entries) and the `E_q(u_-)` closed form for S_RG.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from cone_rgkalgebra import ConeRGKAlgebra
from rgkalgebra import RGKAlgebra
from kalgebra_samples import PentagonKAlg, Sqed1KAlg
from pentagon_bps_iso import _bps_to_pent_label
from habiro import HabiroElement


def _e(terms: dict[tuple, int]) -> Element:
    return Element({k: LaurentPoly({0: v}) for k, v in terms.items()})


# Hard-coded RG image of Pentagon's 5 mult-generators L_i = (i, 1, 0)
# as Sqed1 Elements.  Off-line extraction from
# `SingleNodeRG(BPS(O→O), j=1).RG` + `sqed1_bps_iso`.  The ONLY
# family-specific datum hard-coded for the multiplicative side.
RG_GENERATORS: dict[int, Element] = {
    0: _e({(0,  1): 1}),
    1: _e({(1,  0): 1}),
    2: _e({(1, -1): 1, (0, -1): 1}),
    3: _e({(0, -1): 1, (-1, -1): 1}),
    4: _e({(-1, 0): 1}),
}


def _sqed1_label_to_pent_label(sqed1_label: tuple[int, int]) -> tuple[int, int, int]:
    """`Sqed1` (m, n) ↔ BPS(O→F) charge (n, -m) ↔ Pentagon label via
    `_bps_to_pent_label`'s cone decomposition."""
    m, n = sqed1_label
    return _bps_to_pent_label((n, -m))


class PentagonSquareKAlg(PentagonKAlg, ConeRGKAlgebra):
    """Pentagon presented via cone-data (UV side) + RG flow to Sqed1
    (IR side).  A `ConeRGKAlgebra` instance.

    Inherits all label-structure primitives from `PentagonKAlg`
    (cone-data side: cone_data, coefficient_ring, identity, rho,
    rho_inverse, _label_section_decompose).  RG primitives
    (auxiliary, RG, rg_generator, from_ir_image) are implemented
    below; the `multiply` and `trace` are RG-mediated overrides
    (multiply via `from_ir_image(aux.multiply(RG(a), RG(b)))`;
    trace via Schur transport in Sqed1).

    The class chose RG-mediated multiply rather than cone-data
    multiply for historical reasons (the RG path was implemented
    first); both would produce the same answer on Pentagon's
    canonical basis.  A subclass that prefers the cone-data multiply
    can simply `del PentagonSquareKAlg.multiply` (or inherit without
    the override).
    """

    def __init__(self) -> None:
        PentagonKAlg.__init__(self)
        self._aux = Sqed1KAlg()

    # ----- RGKAlgebra contract --------------------------------------------

    def auxiliary(self) -> Sqed1KAlg:
        return self._aux

    def RG(self, a) -> Element:
        """`RG(L_{i;x,y}) = q^{xy} · RG(L_i)^x · RG(L_{i+1})^y` in Sqed1."""
        i, x, y = a
        if x == 0 and y == 0:
            return Element({self._aux.identity(): LaurentPoly.one()})
        result = Element({self._aux.identity(): LaurentPoly.q(x * y)})
        rg_Li = RG_GENERATORS[i % 5]
        rg_Li1 = RG_GENERATORS[(i + 1) % 5]
        for _ in range(x):
            result = self._aux.multiply_elements(result, rg_Li)
        for _ in range(y):
            result = self._aux.multiply_elements(result, rg_Li1)
        return result

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = E_q(u_-)` truncated to `n ≤ cutoff`."""
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        result: dict[tuple, HabiroElement] = {
            (0, 0): HabiroElement.one(),
        }
        for n in range(1, cutoff + 1):
            sign = -1 if (n % 2 == 1) else 1
            coef = HabiroElement.q_power(n, sign) * HabiroElement.pochhammer_inverse(n)
            result[(-n, 0)] = coef
        return result

    def from_ir_image(self, x_ir: Element) -> Element:
        """Recognise a Sqed1 element as a Pentagon `Element` by tropical
        peel + per-IR-label cone iso via `_sqed1_label_to_pent_label`."""
        result: dict[tuple, LaurentPoly] = {}
        residual = {k: v for k, v in x_ir.terms.items() if not v.is_zero()}
        max_iters = 2000
        iters = 0
        while residual and iters < max_iters:
            iters += 1
            ir_label = max(residual.keys())
            try:
                pent_label = _sqed1_label_to_pent_label(ir_label)
            except ValueError:
                residual.pop(ir_label)
                continue
            rg_pent = self.RG(pent_label)
            if ir_label not in rg_pent.terms:
                raise RuntimeError(
                    f"from_ir_image: ir_label {ir_label} -> pent_label "
                    f"{pent_label}, but RG({pent_label}) doesn't contain "
                    f"{ir_label}; iso inconsistency."
                )
            rg_coef = rg_pent.terms[ir_label]
            res_coef = residual[ir_label]
            items = list(rg_coef._coeffs.items())
            if len(items) != 1 or items[0][1] not in (1, -1):
                raise NotImplementedError(
                    f"from_ir_image: rg_coef {rg_coef} is not ±q^k monomial."
                )
            k, c = items[0]
            alpha = res_coef * LaurentPoly({-k: c})
            result[pent_label] = (
                result.get(pent_label, LaurentPoly.zero()) + alpha
            )
            for lbl, lc in rg_pent.terms.items():
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

    # ----- Overrides: replace PentagonKAlg's multiply and trace ----------

    def multiply(self, a, b):
        """Override `PentagonKAlg.multiply`: use RGKAlgebra's directional
        default (= `from_ir_image(aux.multiply(RG(a), RG(b)))`)."""
        return RGKAlgebra.multiply(self, a, b)

    def trace(self, a, K: int = 20):
        """Override `PentagonKAlg.trace`: Schur transport through Sqed1.

            trace_UV(a)  =  trace_IR( ρ_IR(S_RG) · RG(a) · S_RG )
                         =  inner_product_IR( S_RG, RG(a)·S_RG )

        Best-effort due to S_RG cutoff truncation; uses `cutoff =
        max(K//2, 4)` heuristic + `K_expand = K + cutoff`.
        """
        cutoff = max(K // 2, 4)
        aux = self.auxiliary()
        s_rg = self._s_rg_as_aux_element(cutoff, K + cutoff)
        rg_a = self.RG(a)
        a_S = aux.multiply_elements(rg_a, s_rg)
        rho_s = aux.rho_element(s_rg)
        prod = aux.multiply_elements(rho_s, a_S)
        return aux.trace_element(prod, K)


# Backward-compat alias: the class was renamed from PentagonSquareRGKAlg
# to PentagonSquareKAlg when promoted to ConeRGKAlgebra parent.
PentagonSquareRGKAlg = PentagonSquareKAlg
