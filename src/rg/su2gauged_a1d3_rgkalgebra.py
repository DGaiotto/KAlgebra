"""`SU2GaugedA1D3RGKAlgebra` — **SU(2)-gauged [A₁,D₃]** as the RG flow **to
`U(1)-gauged SU(2) N_f=1`**, integrating out a single **gauge-SU(2)-singlet
monopole** `S_RG = E_𝖖(X_{(1,0)})`.

The **odd rung** of the SU(2)-gauged chain (entry 2 → entry 1) — the SU(2)-gauge
analog of `A1D3Sqed2RGKAlgebra` (`[A₁,D₃] → SQED₂`) and the structural
twin of the odd flavoured rung, gauging the flavour SU(2) of the
A1Dodd–U1A1Deven chain throughout:

    SU(2)-flavoured  :  torus    ← SQED₂(=A1D2)      ← A1D3            ← U1A1D4          ← …
    SU(2)-gauged     :  pureSU2  ← U(1)-gauged-SU2Nf1 ← SU(2)-gauged-A1D3 ← SU(2)×U(1)·A1D4 ← …
                                                        ↑ THIS (→ U(1)-gauged SU(2) N_f=1)

The chain composes: this rung's auxiliary **is** the even-rung flow
`SU2Nf1PureSU2RGKAlgebra` (U(1)-gauged SU(2) N_f=1 over `pure SU(2) ⊗ QT[Z²]`),
so `SU(2)-gauged A1D3 → U(1)-gauged SU(2) N_f=1 → pure SU(2) ⊗ QT[Z²]` chains via
`.then()` down to pure SU(2), the gauge SU(2) dynamical throughout.

Defining data (a **pure** `RGKAlgebra` — generic exact-FS engine, no override)
-----------------------------------------------------------------------------
* `auxiliary()` = `SU2Nf1PureSU2RGKAlgebra` (U(1)-gauged SU(2) N_f=1, the even
  rung).  Spine-free, flavourless (`TrivialZPlusRing` — the SU(2) is gauged).
  Labels `(su2_label, (a, b))`: a pure-SU(2) Wilson label `su2_label` and the
  gauge QT charge `(a, b)`.
* `grading()` = `Γ_RG = Z` = the **magnetic monopole charge `a`** (`X_{ab}·L_c ↦
  a`, `deg(lbl)=lbl[1][0]`) — the `X_{(1,0)}` monopole being integrated out;
  height 1, positive cone `Z_{≥0}`.  (The *odd*-rung magnetic-grading pattern,
  as in `A1Dodd`.)
* `S_RG = E_𝖖(X_{(1,0)})` — `[S_RG]_{(N,)} = {((), X_{(N,0)}): E_𝖖-coeff(N)}`,
  a single **gauge-SU(2)-singlet** monopole tower (the pure-SU(2) Wilson part is
  the trivial rep `w₀ = ()`); exact `HabiroElement` coefficients via
  `sunf_dilog.eq_coeff`.  No
  χ-expansion (contrast the even-rung doublet `E(X₀₁v)E(X₀₁/v)`): the dropped
  hyper is a gauge singlet, the usual single-dilog link.
* `apex` = identity (UV canonical labels are the auxiliary labels).

Validation: the vacuum trace
reproduces the SU(2)-gauged [A₁,D₃] Schur index `1 + q⁸` (matching an
independent BPS-quiver computation of the same index), truncation-stable
(K10 ≡ K14), orthonormal, spine-free.
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
from su2nf1_pure_su2_rgkalgebra import SU2Nf1PureSU2RGKAlgebra
from sunf_dilog import eq_coeff

__all__ = ["SU2GaugedA1D3RGKAlgebra"]


class SU2GaugedA1D3RGKAlgebra(RGKAlgebra):
    """SU(2)-gauged [A₁,D₃] as the pure RG flow `S_RG = E_𝖖(X_{(1,0)})` (a single
    gauge-SU(2)-singlet monopole) to U(1)-gauged SU(2) N_f=1.  The odd rung of
    the SU(2)-gauged chain.  See the module docstring."""

    def __init__(self) -> None:
        self._aux = SU2Nf1PureSU2RGKAlgebra()      # U(1)-gauged SU(2) N_f=1 (even rung)

    # ----- RGKAlgebra defining data --------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = the magnetic X_{(1,0)} charge a = lbl[1][0]; cone Z_{>=0}.
        return Grading(rank=1, deg=lambda lbl: (lbl[1][0],),
                       height=(1,), cone_gens=((1,),))

    def _s_rg_component(self, p) -> dict:
        """`[S_RG]_{(N,)} = {((), X_{(N,0)}): E_𝖖-coeff(N)}` — the single
        gauge-SU(2)-singlet monopole tower `E_𝖖(X_{(1,0)})` (Wilson part the
        trivial rep `w₀ = ()`); `{}` for `N < 0`, the auxiliary identity at
        `N = 0`."""
        (N,) = p
        if N < 0:
            return {}
        if N == 0:
            return {self._aux.identity(): eq_coeff(0)}
        return {((), (N, 0)): eq_coeff(N)}

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = E_𝖖(X_{(1,0)})` windowed to magnetic charge `≤ cutoff`."""
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        out: dict = {}
        for N in range(cutoff + 1):
            out.update(self._s_rg_component((N,)))
        return out

    def _section_split(self, label):
        """Nested `(su2_label, (a,b))` aux labels — the gauge SU(2) is in the
        first tensor factor, not a flat-vector flavour charge — so disable the
        flavour-shift multiply cache (`flav = None`)."""
        return tuple(label), None

    def __repr__(self) -> str:
        return ("SU2GaugedA1D3RGKAlgebra("
                "SU(2)-gauged [A1,D3] → U(1)-gauged SU(2) N_f=1)")


if __name__ == "__main__":
    import warnings
    T = SU2GaugedA1D3RGKAlgebra()
    print(repr(T), " coeff =", T.coefficient_ring())
    print("  aux =", type(T.auxiliary()).__name__, " fs_exact =", T._fs_exact_available())
    print("  S_RG levels 0..3 (single gauge-singlet monopole E(X_{10})):")
    for N in range(4):
        print("    N=%d:" % N, {l: str(c) for l, c in T._s_rg_component((N,)).items()})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vac = T.trace(T.identity(), 10)
        print("  vacuum index (= SU(2)-gauged [A1,D3]):",
              {e: str(r) for e, r in sorted(vac.coeffs.items())}, " warns =", len(w))
