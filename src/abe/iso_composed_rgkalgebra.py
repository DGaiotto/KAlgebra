"""`IsoComposedRGKAlgebra`: post-compose an `RGKAlgebra` with a
`KAlgebraIso` of its auxiliary.

Pattern: given a directional `RGKAlgebra` `inner` (wrapping some
BPSKAlgebra-style IR), and a `KAlgebraIso` between `inner.auxiliary()`
and a standalone IR class, produce a new `RGKAlgebra` whose
`auxiliary()` IS the standalone IR class.  All other directional
data (`RG`, `rg_generator`, `from_ir_image`) is automatically
translated through the iso.

This is the systematic factory for standalone-wrapping RGKAlgebras:

    SingleNodeRGKAlgebra(pentagon-quiver, drop γ_2)
        |  (auxiliary = BPSKAlgebra(O→F))
        |
        +---  KAlgebraIso(BPSKAlgebra(O→F) ↔ Sqed1KAlg)
        |
        v
    IsoComposedRGKAlgebra
        (auxiliary = Sqed1KAlg)
        -- behaves like the hand-rolled PentagonSquareRGKAlg.

Requires the iso to be label-bijective (each label maps to a
single-term Element with coefficient 1).  All BPS-iso family members
on the canonical surface satisfy this.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from rgkalgebra import RGKAlgebra
from kalgebra_iso import KAlgebraIso


def _label_bijective(iso: KAlgebraIso, sample_labels) -> bool:
    """Check (on samples) that `iso.forward_label_map(lbl)` is a
    single-term Element with coefficient 1."""
    for lbl in sample_labels:
        e = iso._forward(lbl)
        if len(e.terms) != 1:
            return False
        (_, c), = e.terms.items()
        if not (c == LaurentPoly.one()):
            return False
    return True


class IsoComposedRGKAlgebra(RGKAlgebra):
    """`RGKAlgebra` whose aux is `inner.auxiliary()` translated through
    a `KAlgebraIso`.

    Constructor:

        IsoComposedRGKAlgebra(inner, aux_iso)

    where:
      * `inner: RGKAlgebra` — the inner directional `RGKAlgebra`.
      * `aux_iso: KAlgebraIso` with `aux_iso.source is inner.auxiliary()`
        (the iso runs `inner.auxiliary() → aux_iso.target`).  The new
        algebra's `auxiliary()` is `aux_iso.target`.

    K-algebra primitives (`multiply`, `rho`, `trace`, …) and the UV
    label set are inherited from `inner`.  RG-side data is translated:
      * `RG(a)` = `aux_iso.map(inner.RG(a))`
      * `rg_generator(cutoff)` translated label-wise via `aux_iso`
        (requires label-bijective iso).
      * `from_ir_image(x_standalone)` = `inner.from_ir_image(
            aux_iso.inverse(x_standalone))`.
    """

    def __init__(self, inner: RGKAlgebra, aux_iso: KAlgebraIso) -> None:
        if aux_iso.source is not inner.auxiliary():
            raise ValueError(
                "IsoComposedRGKAlgebra: aux_iso.source must be (identically) "
                "inner.auxiliary().  Got source = "
                f"{type(aux_iso.source).__name__}, "
                f"inner.auxiliary() = {type(inner.auxiliary()).__name__}.  "
                "If your iso runs the other direction, flip it via a "
                "wrapper that swaps `forward_label_map` and "
                "`inverse_label_map`."
            )
        self._inner = inner
        self._iso = aux_iso
        self._aux = aux_iso.target

    # ----- KAlgebra contract: delegate to inner (UV is the same) ----------

    def coefficient_ring(self):
        return self._inner.coefficient_ring()

    def identity(self):
        return self._inner.identity()

    def multiply(self, a, b):
        # Inherit directional default from RGKAlgebra; but inner already
        # implements multiply (via its own from_ir_image), so we can
        # delegate -- they should agree.  Use the inherited default
        # explicitly so the new aux+from_ir_image path is exercised.
        return super().multiply(a, b)

    def rho(self, a):
        return self._inner.rho(a)

    def rho_inverse(self, a):
        return self._inner.rho_inverse(a)

    def trace(self, a, K=20):
        return self._inner.trace(a, K)

    def _label_section_decompose(self, label):
        return self._inner._label_section_decompose(label)

    # ----- RGKAlgebra contract: translate through aux_iso -----------------

    def auxiliary(self):
        return self._aux

    def RG(self, a) -> Element:
        return self._iso.map(self._inner.RG(a))

    def rg_generator(self, cutoff: int) -> dict:
        """Translate `inner.rg_generator(cutoff)` through `aux_iso`
        (label-wise).  Requires label-bijective iso."""
        inner_dict = self._inner.rg_generator(cutoff)
        result: dict = {}
        for inner_label, habiro_coef in inner_dict.items():
            mapped_elem = self._iso._forward(inner_label)
            if len(mapped_elem.terms) != 1:
                raise NotImplementedError(
                    f"IsoComposedRGKAlgebra.rg_generator: iso is not "
                    f"label-bijective at {inner_label} -> {mapped_elem}; "
                    "general spreading of Habiro coefficients across "
                    "multiple labels not implemented."
                )
            (new_label, c), = mapped_elem.terms.items()
            if not (c == LaurentPoly.one()):
                raise NotImplementedError(
                    f"IsoComposedRGKAlgebra.rg_generator: iso coefficient "
                    f"at {inner_label} -> {new_label} is {c}, not 1."
                )
            result[new_label] = habiro_coef
        return result

    def from_ir_image(self, x_ir: Element) -> Element:
        """Translate `x_ir` (in new aux) back to `inner.aux` via the
        iso's inverse, then call `inner.from_ir_image`."""
        x_inner = self._iso.inverse(x_ir)
        return self._inner.from_ir_image(x_inner)
