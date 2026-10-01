"""`U1A1E7RGKAlgebra` — the **u(1)-gauged `[A₁, E₇]`** Argyres–Douglas algebra
as a **pure** `RGKAlgebra` over `A1A2kKAlg(3) ⊗ QT(Z²)`
(= `[A₁,A₆]` ⊗ a symplectic quantum torus).

The E-type sibling of the A-chain's second leg `U1A1AoddToEvenQTRGKAlgebra` and
of the D-chain's `U1A1DevenSqedRGKAlgebra`: a gauged theory presented as an RG
flow onto `even ⊗ QT`, with the gauged U(1) realised as one leg of the
quantum torus (so no `add_flavour` spectator).

Preferred over the **ungauged** `E7RGKAlgebra` (`[A₁,E₇] → [A₁,A₆] ⊕ U(1)`):
that one carries the U(1) as an `add_flavour(1)` spectator with a heavy
subclass-refined trace (impractically slow). Gauging the U(1) turns it into a
clean quantum-torus leg, putting the flow on the generic **exact-FS** engine —
fast and truncation-safe.

The flow (drop E₇'s branch node, then gauge its U(1))
----------------------------------------------------
The E₇ Dynkin/BPS quiver is the `A₆` chain `1-2-3-4-5-6` with a branch node `7`
off the **trivalent node 4** (arms 3,2,1 from node 4). Dropping the degree-1
node 7 leaves the `A₆` linear quiver — `A1A2kKAlg(3)` — and **gauging node 7's
U(1)** promotes it to one leg of a symplectic `QT(Z²)`:

    auxiliary = A1A2kKAlg(3) ⊗ QT(Z²),   S_RG = E_𝖖(X_{(0,1)} · L_{(3,0)}).

The dressing chord `L = (3, 0)`
-------------------------------
`L` is the chord `E7RGKAlgebra` dresses: the **central** chord `(3, 0)` of the
9-gon.  It crosses the `A₆` chain `(1,0)..(1,5)` of `E7RGKAlgebra._a6_chain`
only at its node 4, and the chain plus `L` has Cartan determinant 2
(`E7RGKAlgebra.uv_cartan_determinant`).  That crossing pattern does **not** pin
the chord: `(2, 4)` and `(3, 4)` also cross the chain only at node 4 (and
`(2, 0)`, `(3, 3)`, `(3, 8)` only at node 3), all with determinant 2, and the
`(2, 4)` dressing gives `Tr(E^{±1}) = 𝖖²`, not the `[A₁,E₇]` value `−𝖖³`
(measured 2026-09-23).  What certifies `(3, 0)` are the traces: the gauged flow
is `E7RGKAlgebra`'s flow with the spectator flavour `μ` made the gauge leg
(measured in the source repository's tests):

    Tr(E^n)             = [μ^{−n}] ((𝖖²;𝖖²)_∞² · Tr_{[A₁,E₇]}(1; μ))   (Nahm e7 vacuum),
    Tr((chord, (0, n))) = [μ^{−n}] ((𝖖²;𝖖²)_∞² · Tr_{E7RGKAlgebra}((chord, (0,)); μ)).

An earlier version dressed the chord `(2, 2)` on the argument that its tropical
c-vector is `−e₄` (node 4).  That chord crosses the chain at nodes 2 and 5, and
the flow it defines is `[A₁,D₇]` with the Cartan `U(1)` of its `SU(2)` flavour
gauged: `Tr(E^{±1}) = 𝖖²` (against `−𝖖³` here), and `Σ_n μ^n Tr(E^n)/(𝖖²;𝖖²)_∞²`
equals the Nahm a1d7 vacuum through `𝖖¹⁰`.  It is kept only as a test-local
negative control.

Labels `(chord, (c0, c1))`: `chord` an `A1A2kKAlg(3)` label, `(c0,c1)` the QT(Z²)
charge; `c1` the dressed/graded leg `(0,1)` (the gauged U(1)), `c0` the
free/magnetic leg `(1,0)`.

Grading (Γ_RG = Z = the gauged-U(1) leg charge)
-----------------------------------------------
`deg((chord, (c0, c1))) = c1`, additive under the auxiliary multiply; positive
cone `Z_{≥0}`, height 1. `apex` is the identity (UV labels = auxiliary labels;
`RG` solved generically).

Engine
------
A **pure** `RGKAlgebra` — `RG` solved, `multiply`/`ρ`/`trace` all the generic
engine, no override. The exact-FS bilinear trace is truncation-safe and fast
(vacuum `1 − 𝖖² + 𝖖⁶ + O(𝖖¹¹)` in about 20 s; the `𝖖²` coefficient is `−1`, the
gauged-U(1) current subtraction). The spine-free auxiliary draws
`A1A2kKAlg(3)` from the cone tier.
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from habiro import HabiroElement
from rgkalgebra import RGKAlgebra
from grading import Grading
from tensor_kalgebra import TensorKAlgebra
from quantum_torus_kalgebra import QuantumTorusKAlg
from a1a2k_kalg import A1A2kKAlg

__all__ = ["U1A1E7RGKAlgebra", "_e_q_coeff"]


def _e_q_coeff(m: int) -> HabiroElement:
    """`c_m = (−q)^m / (q²;q²)_m` — the m-th coefficient of `E_𝖖(X)` for a
    self-pairing-free generator (`⟨e,e⟩ = 0`)."""
    num = LaurentPoly({m: (-1) ** m})
    denom = {j: 1 for j in range(1, m + 1)}
    return HabiroElement(num, denom)


class U1A1E7RGKAlgebra(RGKAlgebra):
    """u(1)-gauged `[A₁, E₇]` over `A1A2kKAlg(3)` (= A₆) ⊗ `QT(Z²)`, dropping
    E₇'s branch node 7 (attached at the trivalent node 4); `S_RG =
    E_𝖖(X_{(0,1)}·L_{(3,0)})`. A pure exact-FS `RGKAlgebra`. See the module
    docstring."""

    # the dressing chord L: the central chord (3, 0), the chord E7RGKAlgebra
    # dresses.  It crosses the A6 chain only at node 4, as (2, 4) and (3, 4) do;
    # the traces, not the crossing pattern, certify it ((2, 4) gives
    # Tr(E^{±1}) = q², and (2, 2) gives [A1,D7] with the Cartan U(1) of its
    # SU(2) gauged; see the module docstring).
    DRESS_TYPE = 3
    DRESS_INDEX = 0

    def __init__(self):
        self._surv = A1A2kKAlg(3)                       # A6 survivor
        self._qt = QuantumTorusKAlg([[0, 1], [-1, 0]])
        self._aux = TensorKAlgebra(self._surv, self._qt)
        self._H = self._surv.H                          # = 9
        self._i0 = self.DRESS_INDEX

    # ----- RGKAlgebra contract -------------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = the dressed QT leg (0,1) charge (the gauged U(1)); the
        # other leg (1,0) is the free/magnetic direction.  Cone Z_{>=0}.
        return Grading(rank=1, deg=lambda lbl: (lbl[1][1],), height=(1,),
                       cone_gens=((1,),))

    def apex(self, a):
        """Identity apex: UV labels coincide with auxiliary labels."""
        return (a[0], tuple(a[1]))

    def _short_chord_power(self, m: int):
        """`L^m` as an `A1A2kKAlg(3)` label: the dressing chord
        `(DRESS_TYPE, DRESS_INDEX) = (3, 0)` to power `m` (it q-commutes with
        itself, so `L^m` is the single label `((3, 0, m),)`)."""
        return ((self.DRESS_TYPE, self._i0, m),)

    def _s_rg_component(self, p):
        """`[S_RG]_{(m,)}` — degree-`m` part (in the dressed leg (0,1)) is the
        single label `(L^m, (0,m))` with coefficient `c_m`."""
        (m,) = p
        if m < 0:
            return {}
        if m == 0:
            return {self._aux.identity(): _e_q_coeff(0)}
        label = (self._short_chord_power(m), (0, m))
        return {label: _e_q_coeff(m)}

    def rg_generator(self, cutoff: int) -> dict:
        out: dict = {}
        for m in range(cutoff):
            out.update(self._s_rg_component((m,)))
        return out

    def _section_split(self, label):
        """The QT charge is a 2-tuple, not a flat-vector charge — fall back to
        the direct `from_ir_image(RG(a)·RG(b))` multiply per pair."""
        return tuple(label), None

    def __repr__(self) -> str:
        return "U1A1E7RGKAlgebra(u(1)-gauged [A1,E7] → [A1,A6] ⊗ QT)"


if __name__ == "__main__":
    import warnings

    def ser(rps, K):
        return {e: str(r) for e, r in sorted(rps.coeffs.items())
                if e <= K and str(r) not in ("0", "")}

    T = U1A1E7RGKAlgebra()
    print(repr(T))
    print("  aux =", type(T.auxiliary()).__name__, "( A1A2kKAlg(3) ⊗ QT(Z²) )  H =", T._H,
          " dressing L = (type", T.DRESS_TYPE, ", i =", T._i0, ") = the central chord")
    print("  exact-FS available:", T._fs_exact_available())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        print("  vacuum (u(1)-gauged E7 index):", ser(T.trace(T.identity(), 10), 10),
              " warns =", len(w))
