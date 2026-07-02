"""`A1D3Sqed2RGKAlgebra` — `A_𝖖([A₁, D₃])` as an RG flow **to SQED₂**
(U(1) gauge + N_f=2, intrinsic SU(2) flavour), dropping a **single
SU(2)-singlet monopole hyper**.

The *same* UV theory `[A₁, D₃]` also admits an RG flow to
**SQED₁ ⊗ SU(2)**.  One UV theory, two distinct RG
flows — they differ entirely in what `S_RG` drops and where the SU(2) lives:

    flow to SQED₁ ⊗ SU(2)_spectator :
                         S_RG = E_𝖖(μL) · E_𝖖(μ⁻¹L)        (an SU(2) DOUBLET
                         of magnetically-charged fork hypers; the SU(2) is the
                         flavour symmetry OF THE DROPPED matter — a spectator).

    A1D3Sqed2RGKAlgebra (this flow, to SQED₂; SU(2) intrinsic, on the
                         electric hypers):
                         S_RG = E_𝖖(u₊)                     (a SINGLE SU(2)-
                         SINGLET monopole hyper; the SU(2) is already the
                         flavour symmetry of SQED₂ — no spectator).

The trap (why "A1D3 → SQED₂" gets stuck)
----------------------------------------
The natural instinct is to copy the SQED₁ recipe: an SU(2)-*charged* `S_RG`
(a doublet `E_𝖖(μL)E_𝖖(μ⁻¹L)`) and/or an externally `add_flavour`-ed SU(2)
spectator.  But **SQED₂ already carries the SU(2)** — it rotates the two
electric hypers (the flavoured monopole relation
`u₊u₋ = 1 + 𝖖·χ_{☐}·v + 𝖖²·v²` over `SUNZPlusRing(2)`).  So the dropped state
is a single SU(2)-**singlet** monopole, `S_RG = E_𝖖(u₊)`, and SQED₂'s own
SU(2) *is* the `[A₁, D₃]` flavour symmetry.  Drop a doublet or bolt on a
spectator and the flavour content double-counts and the index will not close.

Defining data (a **pure** `RGKAlgebra` — generic `RG`/`multiply`/`ρ`/`trace`,
no closed-form override)
------------------------------------------------------------------------------
* `auxiliary()` = `SQEDNfSampleKAlgebra(2)` — SQED₂ in the `u_±/v` presentation,
  labels `(m, n, w)`: magnetic charge `m`, gauge `v`-power `n`, SU(2) irrep `w`
  (an `SUNZPlusRing(2)` partition; `()` = singlet).
* `grading()` = `Γ_RG = Z` = the **magnetic charge** `m` (the monopole power
  `u₊^N = (N, 0, ())`; the electric `v` and the flavour are mag-neutral),
  height 1, positive cone `Z_{≥0}`.  (Contrast `SQEDNfRGKAlgebra`, which grades
  the *same* SQED₂ torus by the **electric** `v`-direction — that flow lands on
  the bare Z² torus; this one grades by the **magnetic** direction and lands on
  SQED₂.)
* `S_RG = E_𝖖(u₊)` — `[S_RG]_{(N,)} = {(N, 0, ()): E_𝖖-coeff(N)}`, a single
  SU(2)-singlet monopole tower (exact `HabiroElement` coefficients via
  `sunf_dilog.eq_coeff`).
* `apex` = identity (UV canonical labels are the auxiliary cone labels).

Validation
----------
The vacuum trace reproduces the `[A₁, D₃]` Schur index **exactly** against
`A1D3KAlg` (q⁰ = singlet, q² = the SU(2) adjoint current χ₂, …) — identical to
the SQED₁ ⊗ SU(2) flow's vacuum, as it must be (same UV theory), and
truncation-stable.  The intrinsic K-algebra axioms (bar-invariance,
RG-multiplicativity, orthonormality) pass on the generic engine.
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rgkalgebra import RGKAlgebra
from grading import Grading
from samples import SQEDNfSampleKAlgebra
from sunf_dilog import eq_coeff

__all__ = ["A1D3Sqed2RGKAlgebra"]


class A1D3Sqed2RGKAlgebra(RGKAlgebra):
    """`[A₁, D₃]` (SU(2) flavour) as the RG flow to **SQED₂** dropping a single
    SU(2)-singlet monopole hyper `S_RG = E_𝖖(u₊)`.  `aux = SQEDNfSampleKAlgebra(2)`,
    graded by the magnetic charge.  A pure `RGKAlgebra`.  See the module docstring."""

    def __init__(self) -> None:
        self._aux = SQEDNfSampleKAlgebra(2)
        self._triv = self._aux.coefficient_ring().one_basis()   # SU(2) singlet

    # ----- RGKAlgebra defining data --------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = magnetic charge m of a SQED₂ label (m, n, w); positive cone
        # Z_{>=0} (the monopole ray u₊^N), height 1.  The electric v-direction n
        # and the SU(2) flavour w are magnetically neutral.
        return Grading(rank=1, deg=lambda lbl: (lbl[0],),
                       height=(1,), cone_gens=((1,),))

    def _s_rg_component(self, p) -> dict:
        """`[S_RG]_{(N,)} = {(N, 0, ()): E_𝖖-coeff(N)}` — the single SU(2)-singlet
        monopole tower `E_𝖖(u₊) = Σ_N E_𝖖-coeff(N)·u₊^N`, `u₊^N = (N, 0, ())`
        (exact in the localized ring); `{}` off the positive cone (`N < 0`)."""
        (N,) = p
        if N < 0:
            return {}
        return {(N, 0, self._triv): eq_coeff(N)}

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = E_𝖖(u₊)` windowed to level ≤ `cutoff` (the level-`N`
        coefficient carries leading q-order `N`)."""
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        out: dict = {}
        for N in range(cutoff + 1):
            out.update(self._s_rg_component((N,)))
        return out

    def __repr__(self) -> str:
        return "A1D3Sqed2RGKAlgebra([A1,D3] → SQED2, single SU(2)-singlet monopole)"


if __name__ == "__main__":
    import warnings

    A = A1D3Sqed2RGKAlgebra()
    print(repr(A))
    print("  aux =", repr(A.auxiliary()), "  coeff ring =", A.coefficient_ring())
    print("  exact-FS available:", A._fs_exact_available())
    print("  S_RG = E_q(u+) levels 0..3:")
    for N in range(4):
        for lab, c in A._s_rg_component((N,)).items():
            print(f"    N={N}: {lab}  coeff={c}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vac = A.trace(A.identity(), 6)
        print("  vacuum Schur index (= [A1,D3]):",
              {e: str(r) for e, r in sorted(vac.coeffs.items()) if str(r) not in ("0", "")},
              " warns =", len(w))
