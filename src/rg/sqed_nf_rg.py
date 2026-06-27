"""`SQEDNfRGKAlgebra` — `SQED_{N_f}` (U(1) gauge + `N_f` charged hypers, **SU(N_f)
flavour**) presented as an RG flow to the Z² quantum torus.  The general
`N_f` flavoured analogue of `U1SquareRGKAlgebra` (N_f=1) and `U1A1D2RGKAlgebra`
(N_f=2), and a **pure** `RGKAlgebra`: `RG`, `multiply`, `ρ`, `trace` are all the
generic engine — no closed-form override.

Defining data
-------------
* `auxiliary()` = `Z2QTorusSampleKAlgebra().add_flavour(SUNZPlusRing(N_f))` — the
  sample Z² quantum torus with an **SU(N_f)** flavour adjoined (the augmented-ring
  route; the sample torus is otherwise untouched).  Labels `((a,b), w)`: a QT
  charge `(a,b) ∈ Z²` dressed by an SU(N_f) irrep `w` (a partition).
* `grading()` = `Γ_RG = Z`, `deg((a,b),w) = b` (the `X_{0,1}`-multiplicity, i.e.
  the gauge magnetic charge the matter carries), height `b`, positive cone
  `Z_{≥0}`.
* `S_RG = ∏_{i=1}^{N_f} E_𝖖(μ_i·X_{0,1})` pushed through the Weyl-symmetric
  `μ → χ_{SU(N_f)}` map — the `N_f` collinear hypers form an SU(N_f) fundamental.
  The exact per-level SU(N_f)-character content is `sunf_dilog.sunf_components`
  (the bialternant Weyl peel, Habiro-exact); `_s_rg_component((N,))` wraps it at
  the carrier label `X_{0,N} = (0, N)`.
* `apex` = identity (UV canonical labels are the auxiliary cone labels).

Why it is fully generic
-----------------------
The trace is the engine's **bilinear pairing** over the SU(N_f)-flavoured
auxiliary (the per-charge `RG(a)·S_RG` components are exact via the nested-aux
exact-FS walk, #666; the flavour is handled by `aux.inner_product`).  `multiply`
is the generic `from_ir_image(RG(a)·RG(b))`: the SU(N_f) flavour is **non-additive**
(it fuses by Clebsch–Gordan, not by a central shift), so the generic `multiply`
takes its non-additive-flavour branch and lets `aux.multiply` do the fusion — no
subclass `multiply`/`trace` override is needed (unlike the `A1Dodd`-derived
`U1A1D2`, which carries a heavier closure).

`SQEDNfRGKAlgebra(1)` reproduces SQED₁ (= `U1SquareRGKAlgebra`);
`SQEDNfRGKAlgebra(2)` reproduces SQED₂ (= `U1A1D2RGKAlgebra`).  Certified by a
`KAlgebraIso` to `SQEDNfSampleKAlgebra(N_f)` (the same `u_±/v` presentation),
relabelling `((a,b), w) ↔ (a, b, w)` — both sides over `SUNZPlusRing(N_f)`.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rgkalgebra import RGKAlgebra
from grading import Grading
from zplus_ring import SUNZPlusRing
from samples import Z2QTorusSampleKAlgebra
from sunf_dilog import sunf_components, eq_coeff

__all__ = ["SQEDNfRGKAlgebra"]


class SQEDNfRGKAlgebra(RGKAlgebra):
    """`SQED_{N_f}` as the SU(N_f)-flavoured RG flow
    `S_RG = ∏_i E_𝖖(μ_i X_{0,1}) → χ_{SU(N_f)}` to the Z² quantum torus.  A pure
    `RGKAlgebra`; `aux = Z2QTorusSampleKAlgebra().add_flavour(SUNZPlusRing(N_f))`."""

    def __init__(self, Nf: int) -> None:
        if Nf < 1:
            raise ValueError(f"SQEDNfRGKAlgebra: N_f >= 1 required, got {Nf}")
        self._Nf = int(Nf)
        self._base = Z2QTorusSampleKAlgebra()
        self._aux = self._base.add_flavour(SUNZPlusRing(self._Nf))

    # ----- RGKAlgebra defining data --------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        return Grading(
            rank=1,
            deg=lambda lbl: (lbl[0][1],),       # deg(((a,b),w)) = b
            height=(1,),                        # h(b) = b
            cone_gens=((1,),),                  # positive cone = Z_{≥0}
        )

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = ∏_i E_𝖖(μ_i X_{0,1})` windowed to q-order ≤ `cutoff` (the
        level-`N` SU(N_f)-character content carries leading q-order ≥ `N`)."""
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        out: dict = {}
        for N in range(cutoff + 1):
            for w, c in sunf_components(self._Nf, N).items():
                out[((0, N), w)] = c
        return out

    def _s_rg_component(self, p) -> dict:
        """Exact `Γ_RG`-graded component `[S_RG]_{(N,)}`: the SU(N_f)-irrep
        content of `[x^N] ∏_i E_𝖖(μ_i x)` at the carrier label `X_{0,N} = (0,N)`
        — `{((0,N), w): coeff}` for `N ≥ 0`, `{}` off the positive cone."""
        (N,) = p
        if N < 0:
            return {}
        return {((0, N), w): c for w, c in sunf_components(self._Nf, N).items()}

    def __repr__(self) -> str:
        return f"SQEDNfRGKAlgebra({self._Nf})"
