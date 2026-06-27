"""`SU2U1GaugedA1D4RGKAlgebra` — **SU(2)×U(1)-gauged [A₁,D₄]** as the RG flow **to
`SU(2)-gauged A1D3 ⊗ QT[Z²]`**, integrating out a single **gauge-SU(2)-doublet**
matter hyper `S_RG = E_𝖖(X₀₁v)·E_𝖖(X₀₁v⁻¹)`.

The **even rung** of the SU(2)-gauged chain (entry 3 → entry 2) — the
`+QT[Z²]` matter rung one rank above the base `SU2Nf1PureSU2RGKAlgebra` (entry 1),
gauging the flavour SU(2) of the A1Dodd–U1A1Deven chain throughout (user,
2026-06-27):

    SU(2)-flavoured  :  …  ← A1D3        ← U1A1D4            ← A1D5 ← …
    SU(2)-gauged     :  …  ← SU(2)·A1D3  ← SU(2)×U(1)·A1D4   ← …
                                          ↑ THIS  (→ SU(2)-gauged A1D3 ⊗ QT[Z²])

The chain composes: this rung's auxiliary **is** `SU2GaugedA1D3RGKAlgebra` (entry
2) ⊗ a fresh `QT[Z²]`, so the whole chain `… → entry 2 → entry 1 → pure SU(2)`
chains via `.then()`, the gauge SU(2) dynamical throughout.

Defining data (a **pure** `RGKAlgebra` — generic exact-FS engine, no override)
-----------------------------------------------------------------------------
* `auxiliary()` = `SU2GaugedA1D3RGKAlgebra (entry 2 flow) ⊗
  QuantumTorusKAlg([[0,1],[-1,0]])` (the fresh gauge torus).  Spine-free,
  flavourless.  Labels `((su2_label, (a₂,b₂)), (a,b))`: the entry-2 label (a
  pure-SU(2) Wilson + entry-2's QT charge) ⊗ the fresh QT charge.
* `grading()` = `Γ_RG = Z` = the **fresh gauge leg charge `b`** (`X_{ab}·L_c ↦ b`,
  `deg = lbl[1][1]`) — the even-rung gauge grading, as in entry 1.
* `S_RG = E_𝖖(X₀₁v)·E_𝖖(X₀₁v⁻¹)` — the two-weight gauge doublet on the fresh
  electric leg, χ-expanded with `χ_n → w_n` (the pure-SU(2) Wilson reached
  through the nested aux as `((w_n, (0,0)), (0,N))`):

      [S_RG]_{(N,)} = Σ_{n ≡ N (2), 0 ≤ n ≤ N}  d_n(N) · ((w_n,(0,0)), X_{(0,N)}),
      d_n(N) = c_{(N+n)/2}c_{(N-n)/2} − c_{(N+n)/2+1}c_{(N-n)/2-1}   (c_m = E_𝖖-coeff, 0 for m<0)

  — the identical even-rung recipe as `SU2Nf1PureSU2RGKAlgebra`, one level up.
* `apex` = identity.

Validation: `Tr(1) = 1 − q² − q⁴ +
q¹² + 2q¹⁴ + …`, truncation-stable, matching the **BPS oracle** — the U(1)-Haar
of the SU(2)-gauged A1D4 BPSKAlgebra, `(q²;q²)_∞² · μ⁰[su2a1d3_gauged.su2a1d4_det1(μ)]`
(the N=2\*-paper gauging trick).  `w₁² = 1 + w₂`, orthonormal, spine-free.
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
from tensor_kalgebra import TensorKAlgebra
from quantum_torus_kalgebra import QuantumTorusKAlg
from su2gauged_a1d3_rgkalgebra import SU2GaugedA1D3RGKAlgebra
from su2nf1_pure_su2_rgkalgebra import _w, _c
from sunf_dilog import eq_coeff

__all__ = ["SU2U1GaugedA1D4RGKAlgebra"]


class SU2U1GaugedA1D4RGKAlgebra(RGKAlgebra):
    """SU(2)×U(1)-gauged [A₁,D₄] as the pure RG flow `S_RG = E_𝖖(X₀₁v)E_𝖖(X₀₁v⁻¹)`
    (the gauge doublet) to `SU(2)-gauged A1D3 ⊗ QT[Z²]`.  The even rung of the
    SU(2)-gauged chain.  See the module docstring."""

    def __init__(self) -> None:
        self._e2 = SU2GaugedA1D3RGKAlgebra()                  # entry 2 flow
        self._qt = QuantumTorusKAlg([[0, 1], [-1, 0]])        # fresh gauge torus
        self._aux = TensorKAlgebra(self._e2, self._qt)

    # ----- RGKAlgebra defining data --------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = the fresh gauge leg charge b = lbl[1][1]; cone Z_{>=0}.
        return Grading(rank=1, deg=lambda lbl: (lbl[1][1],),
                       height=(1,), cone_gens=((1,),))

    def _wlabel(self, n: int):
        """The pure-SU(2) Wilson `w_n` as an entry-2-flow label: `(w_n, (0,0))`
        (gauge-neutral in entry-2's own QT charge)."""
        return (_w(n), (0, 0))

    def _s_rg_component(self, p) -> dict:
        """`[S_RG]_{(N,)}` — the gauge doublet `E_𝖖(X₀₁v)E_𝖖(X₀₁v⁻¹)` at fresh
        gauge charge `N`, χ-expanded with `χ_n → w_n` (reached through the nested
        aux):

            Σ_{n ≡ N (2), 0 ≤ n ≤ N}  d_n(N) · ((w_n,(0,0)), X_{(0,N)}),
            d_n(N) = c_{(N+n)/2}·c_{(N-n)/2} − c_{(N+n)/2+1}·c_{(N-n)/2-1}.

        `{}` for `N < 0`; the auxiliary identity at `N = 0`."""
        (N,) = p
        if N < 0:
            return {}
        if N == 0:
            return {self._aux.identity(): eq_coeff(0)}
        out: dict = {}
        for n in range(N, -1, -2):
            a, b = (N + n) // 2, (N - n) // 2
            term = _c(a) * _c(b)
            lo, hi = _c(a + 1), _c(b - 1)
            if lo is not None and hi is not None:
                term = term - lo * hi
            if not term.is_zero():
                out[(self._wlabel(n), (0, N))] = term
        return out

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = E_𝖖(X₀₁v)E_𝖖(X₀₁v⁻¹)` windowed to fresh gauge charge `≤ cutoff`."""
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        out: dict = {}
        for N in range(cutoff + 1):
            out.update(self._s_rg_component((N,)))
        return out

    def _section_split(self, label):
        """Doubly-nested `((su2_label,(a₂,b₂)),(a,b))` aux labels — the gauge
        SU(2) is the deepest tensor factor — so disable the flavour-shift multiply
        cache (`flav = None`)."""
        return tuple(label), None

    def __repr__(self) -> str:
        return ("SU2U1GaugedA1D4RGKAlgebra("
                "SU(2)×U(1)-gauged [A1,D4] → SU(2)-gauged A1D3 ⊗ QT[Z²])")


if __name__ == "__main__":
    import warnings
    T = SU2U1GaugedA1D4RGKAlgebra()
    print(repr(T), " coeff =", T.coefficient_ring())
    print("  aux =", type(T.auxiliary()).__name__, " fs_exact =", T._fs_exact_available())
    print("  S_RG levels 0..3 (gauge doublet over entry-2 ⊗ QT):")
    for N in range(4):
        print("    N=%d:" % N, {l: str(c) for l, c in T._s_rg_component((N,)).items()})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vac = T.trace(T.identity(), 6)
        print("  Tr(1):", {e: str(r) for e, r in sorted(vac.coeffs.items())}, " warns =", len(w))
