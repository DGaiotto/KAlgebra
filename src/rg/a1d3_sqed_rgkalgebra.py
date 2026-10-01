"""`A1D3SqedRGKAlgebra` — `A_𝖖([A₁, D₃])` as an **SU(2)-flavoured-native
RGKAlgebra flowing to SQED1**.

This is the **base case (`k = 0`)** of the odd-`D` family
`A1DoddRGKAlgebra` (`[A₁, D_{2k+3}]`, `k ≥ 1 → D₅, D₇, …`).  At `k = 0`
the IR survivor `U1A1AoddKAlg(0)` *is* the U(1)-gauged square = **SQED1**
(`U1SquareKAlg`), so A1D3 = `[A₁, D₃]` is the RG flow

    auxiliary  =  U1SquareKAlg().add_flavour(SU2ZPlusRing())     (= SQED1 ⊗ SU(2))
    S_RG       =  E_𝖖(μ·L) · E_𝖖(μ⁻¹·L)

where `L` is SQED1's magnetic-charge-1 chord (the monopole `(1,0)`).  The
two collinear quantum dilogs are the two fork hypers forming an **SU(2)
doublet** (weights `μ^{±1}`); over the gauged-`U(1)` survivor they combine
into honest **SU(2) characters** (the χ-peel `c_a c_b − c_{a+1} c_{b−1}`),
so the coefficient ring is `SU2ZPlusRing` *natively* — no U(1)-Cartan
fugacity, no after-the-fact base change.  This is the SU(2)-native
counterpart of the `DirectionalSubquiverRG` node-drop (which drops the
two SU(2)-flavoured `D₃` leaves but lands in the U(1)-Cartan frame).

Construction
------------
A thin subclass of `A1DoddRGKAlgebra` overriding only the three
base-specific hooks — `__init__` (the SQED1 base), `_mag` (the magnetic
charge of a `(m, n)` label is `m`), and `_short_chord_power` (`L^N =
(N, 0)`).  Everything else — the χ-peel `_s_rg_component`, the
graded `RG(a)·S_RG` assembly, the SU(2)-refined `inner_product`/`trace`,
the generic `RG`/`multiply`/`ρ` — is inherited unchanged from
`A1DoddRGKAlgebra` / `RGKAlgebra`.

Canonical labels are the auxiliary cone monomials `((m, n), κ)` (a SQED1
`(m, n)` label tensored with the SU(2) highest weight `κ`); the apex is
the identity (UV labels = auxiliary labels).

Validation: the vacuum trace reproduces the `[A₁, D₃]` Schur index
*exactly* against `A1D3KAlg` (the standalone Layer-2 closed form) — q⁰ =
singlet, q² = the SU(2) adjoint current χ₂ — and the intrinsic K-algebra
axioms pass.  See the suite in the source repository.

The `(2k+4)`-gon geometric-label helpers of `A1DoddRGKAlgebra` do not
apply at the square base (`U1SquareKAlg` carries no `(2k+4)`-gon
`geometric_label`), so they are disabled here.
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from zplus_ring import SU2ZPlusRing
from u1_square_kalg import U1SquareKAlg
from a1dodd_rgkalgebra import A1DoddRGKAlgebra

__all__ = ["A1D3SqedRGKAlgebra"]


class A1D3SqedRGKAlgebra(A1DoddRGKAlgebra):
    """`[A₁, D₃]` (SU(2) flavour) as the SQED1 base case of the A1Dodd
    RG family.  `aux = U1SquareKAlg().add_flavour(SU2)`,
    `S_RG = E_𝖖(μL)E_𝖖(μ⁻¹L)` with `L` the SQED1 monopole `(1,0)`."""

    def __init__(self):
        self.k = 0
        self._base = U1SquareKAlg()                       # SQED1 (= U1A1Aodd(0))
        self._aux = self._base.add_flavour(SU2ZPlusRing())

    # ----- base-specific hooks (the only overrides) ----------------------

    def _mag(self, cone_label) -> int:
        """Magnetic charge of a SQED1 `(m, n)` label is `m` (the L-power;
        the gauge/`v` direction `n` is magnetically neutral)."""
        return cone_label[0]

    def _short_chord_power(self, N: int):
        """`L^N` as a SQED1 label: the magnetic-`N` monopole `(N, 0)`
        (the identity `(0, 0)` for `N = 0`)."""
        if N == 0:
            return self._base.identity()                  # (0, 0)
        return (N, 0)

    # ----- the (2k+4)-gon geometry does not apply at the square base -----

    def _hexchord(self, ray):
        raise NotImplementedError(
            "A1D3SqedRGKAlgebra: the (2k+4)-gon geometric labels do not "
            "apply at the SQED1 (square) base case.")

    def geometric_label(self, ray):
        raise NotImplementedError(
            "A1D3SqedRGKAlgebra: the (2k+4)-gon geometric labels do not "
            "apply at the SQED1 (square) base case.")

    def intersection_number(self, ray_a, ray_b):
        raise NotImplementedError(
            "A1D3SqedRGKAlgebra: the (2k+4)-gon geometric labels do not "
            "apply at the SQED1 (square) base case.")

    def __repr__(self) -> str:
        return "A1D3SqedRGKAlgebra([A1,D3] → SQED1, SU(2)-native)"
