"""
a1aeven_to_u1aodd_rgkalgebra.py
===============================

`A1AevenToU1AoddRGKAlgebra(k)` — the even Argyres–Douglas family
`A_𝖖([A_1, A_{2k+2}])` as an **`RGKAlgebra`**,
realised by the RG flow

    [A_1, A_{2k+2}]   ──drop first node──▶   u(1)-gauged [A_1, A_{2k+1}]

i.e. the UV is the even AD theory and the IR is the *gauged-odd* theory
one level down.  This is the gauged-IR companion of the ungauged
`a1aodd_to_even_rgkalgebra.A1AoddToEvenRGKAlgebra` (odd UV → even IR):
here the survivor of the single-node drop is itself U(1)-gauged, so the
auxiliary is the standalone closed-form class

    auxiliary  =  U1A1AoddKAlg(k)              # u(1)-gauged [A_1, A_{2k+1}]

*directly* — no `add_flavour`, no quantum-torus tensor.  The dropped
node's direction is already present inside `U1A1AoddKAlg(k)` as the
**magnetic** charge of the gauge U(1) (the gauge generator `E` itself is
magnetically neutral, `mag(E)=0`).

The flow (drop the first node of `O>O>…>O`)
-------------------------------------------
`[A_1, A_{2k+2}]` has the linear `A_{2k+2}` BPS quiver
`γ₁—γ₂—⋯—γ_{2k+2}`.  Dropping the terminal node γ₁ leaves the surviving
chain as the *gauged* `[A_1, A_{2k+1}]`: γ₁ pairs only with γ₂, which is
the magnetic partner of the gauge U(1), so the dropped node integrates
out as the magnetic-charge tower rather than a spectator flavour
(contrast the central, commuting drop in the ungauged sibling — there
the terminal of the *odd* quiver is the lattice kernel μ).

Grading (Γ_RG = Z = the magnetic charge)
----------------------------------------
The grading lattice is **Z** — the magnetic charge `mag` of the gauge
U(1), read off the cone label `(factors, e_E)` as
`mag = Σ_factors exp · mag(chord)` (the gauge generator `E` is
mag-neutral, so `e_E` does not contribute).  Positive cone `Z_{≥0}`,
height 1.  `apex` is the identity: the UV canonical labels are taken to
be the auxiliary cone labels, and `RG(a)` is solved from the discovery
relation `RG(a)·S_RG = L_a + O(q)`.

Spectrum generator
------------------
    S_RG  =  E_𝖖(X_L)

with `L` a **short chord ray of magnetic charge 1** in `U1A1AoddKAlg(k)`
(a length-2 (2k+4)-gon diagonal).  Since `L`
q-commutes with itself (`⟨L,L⟩ = 0`), its m-th power is the single cone
monomial `L^m = (((1, i₀, m),), 0)` (coefficient 1, no E-drift —
verified), so the exact graded component is

    [S_RG]_{(m,)}  =  c_m · (((1, i₀, m),), 0),   c_m = (−q)^m / (q²;q²)_m,

a single auxiliary label per magnetic degree; degree 0 is the identity.

Validation
----------
The constructed UV algebra agrees with the standalone
even class `A1A2kKAlg(k+1) = A_𝖖([A_1, A_{2k+2}])` (and the genuine BPS
realisation over the `A_{2k+2}` chain) on structure constants, via the
shared `A_{2k+2}`-lattice charge map.  The form of the flow (`S_RG`, `RG`)
cross-checks against the generic single-node drop (`SingleNodeRG`, in the
BPS layer).
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from habiro import HabiroElement
from rgkalgebra import RGKAlgebra
from grading import Grading
from u1a1aodd_kalg import U1A1AoddKAlg


def _e_q_coeff(m: int) -> HabiroElement:
    """`c_m = (−q)^m / (q²;q²)_m` — the m-th coefficient of `E_𝖖(X)` for a
    self-pairing-free generator (`⟨L,L⟩ = 0`).  As a `HabiroElement`: numerator
    `(−1)^m q^m`, denominator `∏_{j=1}^{m}(1 − q^{2j})`."""
    num = LaurentPoly({m: (-1) ** m})
    denom = {j: 1 for j in range(1, m + 1)}
    return HabiroElement(num, denom)


class A1AevenToU1AoddRGKAlgebra(RGKAlgebra):
    """`[A_1, A_{2k+2}]` as a directional `RGKAlgebra` wrapping
    the gauged-odd standalone `U1A1AoddKAlg(k)`, with `RG` from the exact
    per-charge solver.  See the module docstring."""

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._aux = U1A1AoddKAlg(k)
        self._cd = self._aux.cone_data()
        self._n = self._cd._n                       # = 2k + 2 (B_GAUGED lattice rank)
        self._mag_index = self._n - 1               # the F coord = the magnetic charge
        # dressing chord: a short chord (type a=1) of magnetic charge +1.
        chg = self._cd._chg
        shorts_mag1 = sorted(
            i for i in self._cd._types[1] if chg[(1, i)][self._mag_index] == 1
        )
        if not shorts_mag1:
            raise RuntimeError(f"no magnetic-charge-1 short chord at k={k}")
        self._i0 = shorts_mag1[0]

    # ----- magnetic grading helper ---------------------------------------

    def _mag(self, label) -> int:
        """Magnetic charge of a cone label `(factors, e_E)` — `Σ exp·mag(chord)`
        over the chord factors (`E` is mag-neutral, so `e_E` drops out)."""
        chg, mi = self._cd._chg, self._mag_index
        factors, _e_E = label
        return sum(exp * chg[(a, i)][mi] for (a, i, exp) in factors)

    # ----- RGKAlgebra contract -------------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = the magnetic charge; positive cone Z_{>=0}, height 1.
        return Grading(rank=1, deg=lambda lbl: (self._mag(lbl),), height=(1,),
                       cone_gens=((1,),))

    def apex(self, a):
        """Identity apex: UV labels coincide with auxiliary cone labels."""
        a = tuple(a)
        return (a[0], a[1])

    def _short_chord_power(self, m: int):
        """`L^m` as a `U1A1AoddKAlg` cone label: the mag-1 short dressing
        chord raised to `m` (a single q-commuting generator power)."""
        return (((1, self._i0, m),), 0)

    def _s_rg_component(self, p):
        """`[S_RG]_{(m,)}` — exact, finite, vanishing off the cone.

        `S_RG = E_𝖖(X_L)` ⇒ degree-`m` part (in magnetic charge) is the
        single label `L^m` with coefficient `c_m`; degree 0 is the
        auxiliary identity, negative degree empty."""
        (m,) = p
        if m < 0:
            return {}
        if m == 0:
            return {self._aux.identity(): _e_q_coeff(0)}
        return {self._short_chord_power(m): _e_q_coeff(m)}

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG` windowed to q-order ≤ `cutoff`: the magnetic tower
        `{L^m : c_m}` for `m < cutoff` (the exact `RG` path uses
        `_s_rg_component`)."""
        out: dict = {}
        for m in range(cutoff):
            out.update(self._s_rg_component((m,)))
        return out

    # ----- flavour-aware section split -----------------------------------

    def _section_split(self, label):
        """The auxiliary cone label `(factors, e_E)` is not a flat charge
        vector, so the generic componentwise `_section_split` does not apply.
        Disable the flavour-shift multiply cache (`flav = None`): `multiply`
        falls back to the direct `from_ir_image(RG(a)·RG(b))` per pair."""
        return tuple(label), None


if __name__ == "__main__":
    T = A1AevenToU1AoddRGKAlgebra(1)
    print("k=1: aux =", type(T.auxiliary()).__name__,
          " n =", T._n, " dressing short-chord i0 =", T._i0,
          " (mag =", T._mag(T._short_chord_power(1)), ")")
    print("identity =", T.identity())
    a = (((2, 0, 1),), 0)          # a long (mag-0) chord of U1A1AoddKAlg(1)
    print("RG(L(2,0)) =", {l: str(c) for l, c in T.RG(a).terms.items()})
    b = (((2, 1, 1),), 0)
    print("multiply(L(2,0), L(2,1)) =",
          {l: str(c) for l, c in T.multiply(a, b).terms.items()})
