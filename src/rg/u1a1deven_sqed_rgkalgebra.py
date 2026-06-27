"""`U1A1DevenSqedRGKAlgebra(k)` — the u(1)-gauged `[A₁, D_{2k+2}]` as the RG flow
**to `A1Dodd(k-1) ⊗ QT[Z²]`** dropping a single gauge-dressed doublet hyper
`S_RG = E_𝖖(X_{(0,1)}·L)`.  The even-rung companion of `A1D3Sqed2RGKAlgebra`
(#675) in the A1Dodd–U1A1Deven chain.

    …  ←  D2 (SQED2)  ←  D3 (A1D3)  ←  D4 (U1A1D4)  ←  D5 (A1D5)  ← …
                          ↑ #675          ↑ this (k=1)

The ladder alternates: the **odd** rungs `A1Dodd(k)` are SU(2)-flavoured AD
theories (each a single SU(2)-singlet monopole drop, `A1D3Sqed2RGKAlgebra`); the
**even** rungs `U1A1Deven(k)` are their **u(1)-gauged** siblings — gauging the
odd survivor `A1Dodd(k-1) = [A₁, D_{2k+1}]` by a fresh `QT[Z²]` and integrating
out a single **gauge-dressed doublet** hyper.  `k = 1` is **U1A1D4** (gauged
`[A₁, D₄]`) over **A1D3 ⊗ QT[Z²]**.

Defining data (a **pure** `RGKAlgebra` — generic `RG`/`multiply`/`ρ`/`trace`,
no closed-form override)
------------------------------------------------------------------------------
* `auxiliary()` = `A1DoddConeKAlg(k-1) ⊗ QT([[0,1],[-1,0]])` — the spine-free
  odd survivor `[A₁, D_{2k+1}]` (SU(2) flavour **intrinsic**, coeff
  `SUNZPlusRing`/`SU2ZPlusRing`) tensored with the fresh rank-2 gauge torus.
  Labels `(dodd_label, (c0, c1))`: `dodd_label = (word, κ)` an A1Dodd cone
  label, `(c0, c1)` the QT charge.  Since the SU(2) is in the survivor, the
  dropped hyper carries it intrinsically — **no `add_flavour` spectator, no
  doublet `S_RG`** (contrast the legacy `A1DevenRGKAlgebra`, whose
  `add_flavour(SU2×U1)` + `E(μ₁L)E(μ₂L)` double-counts the flavour).
* `grading()` = `Γ_RG = Z` = the **gauge** `c1`-leg charge `X_{(0,1)}` (the
  monopole power being integrated out; height 1, positive cone `Z_{≥0}`).
* `S_RG = E_𝖖(X_{(0,1)}·L)` — degree-`N` part `(L^N, (0, N))` with the Habiro
  `E_𝖖` coefficient, `L = (k, 1, i)` the **length-`k` doublet** (p=1, χ₁) chord
  of `A1Dodd(k-1)`: its A1Dodd trace carries the half-integer spins, so the
  single-letter `E_𝖖(X₀₁·L)` reproduces the legacy doublet
  `E(μX₀₁L)E(μ⁻¹X₀₁L)` folding intrinsically (the `u1a1deven_via_dodd_rg`
  validation: length-`k` is exact, shorter doublets over-count from q^{2k+2}).
* `apex` = identity (UV canonical labels are the auxiliary cone labels).

This is the spine-free / generic-engine finalisation of
`u1a1deven_via_dodd_rg.U1A1DevenViaDoddRG` (which carries a legacy *windowed*
trace override): here `_fs_exact_available()` is `True` (post-#666 the exact-FS
bilinear trace fires over the nested tensor aux), so the trace is the generic
**truncation-safe bilinear** pairing with **no override**.
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
from a1dodd_kalg import A1DoddConeKAlg
from sunf_dilog import eq_coeff

__all__ = ["U1A1DevenSqedRGKAlgebra"]


class U1A1DevenSqedRGKAlgebra(RGKAlgebra):
    """u(1)-gauged `[A₁, D_{2k+2}]` as the pure RG flow
    `S_RG = E_𝖖(X_{(0,1)}·L)` to `A1Dodd(k-1) ⊗ QT[Z²]`.  `k = 1` is U1A1D4 over
    A1D3 ⊗ QT[Z²].  See the module docstring."""

    def __init__(self, k: int = 1, chord=None) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._surv = A1DoddConeKAlg(k - 1)                  # [A₁, D_{2k+1}] = A1Dodd(k-1)
        self._qt = QuantumTorusKAlg([[0, 1], [-1, 0]])      # fresh rank-2 gauge torus
        self._aux = TensorKAlgebra(self._surv, self._qt)
        # L = the length-k DOUBLET chord (k,1,i) of A1Dodd(k-1) (any i; default 0).
        self._L = (k, 1, 0) if chord is None else tuple(chord)

    # ----- RGKAlgebra defining data --------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = the gauge c1-leg (0,1) charge; positive cone Z_{>=0}, height 1.
        return Grading(rank=1, deg=lambda lbl: (lbl[1][1],),
                       height=(1,), cone_gens=((1,),))

    def _dodd_label_pow(self, m: int):
        """`L^m` as an `A1Dodd(k-1)` cone label `(word, κ=0)`: the chord `L` to
        power `m` (`word = ((L, m),)`; the identity `((), 0)` for `m = 0`)."""
        if m == 0:
            return self._surv.identity()
        return (((self._L, m),), 0)

    def _s_rg_component(self, p) -> dict:
        """`[S_RG]_{(N,)} = {(L^N, (0, N)): E_𝖖-coeff(N)}` — the single
        gauge-dressed doublet monopole tower `E_𝖖(X_{(0,1)}·L)` (Habiro-exact);
        `{}` off the positive cone (`N < 0`), the auxiliary identity at `N = 0`."""
        (N,) = p
        if N < 0:
            return {}
        if N == 0:
            return {self._aux.identity(): eq_coeff(0)}
        return {(self._dodd_label_pow(N), (0, N)): eq_coeff(N)}

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = E_𝖖(X_{(0,1)}·L)` windowed to level ≤ `cutoff`."""
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        out: dict = {}
        for N in range(cutoff + 1):
            out.update(self._s_rg_component((N,)))
        return out

    def _section_split(self, label):
        """Nested `(dodd_label, (c0,c1))` aux labels — the flavour is in the
        survivor's κ, not a flat-vector charge — so disable the flavour-shift
        multiply cache (`flav = None`)."""
        return tuple(label), None

    def __repr__(self) -> str:
        return f"U1A1DevenSqedRGKAlgebra(k={self.k}, [A1,D{2*self.k+2}] → A1Dodd({self.k-1})⊗QT)"


if __name__ == "__main__":
    import warnings
    T = U1A1DevenSqedRGKAlgebra(1)
    print(repr(T), " coeff =", T.coefficient_ring())
    print("  aux =", type(T.auxiliary()).__name__, " surv =", type(T._surv).__name__, " L =", T._L)
    print("  _fs_exact_available:", T._fs_exact_available())
    print("  S_RG levels 0..2:")
    for N in range(3):
        print("    N=%d:" % N, {l: str(c) for l, c in T._s_rg_component((N,)).items()})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vac = T.trace(T.identity(), 4)
        print("  vacuum index (= U1A1D4 Schur index):",
              {e: str(r) for e, r in sorted(vac.coeffs.items())}, " warns =", len(w))
