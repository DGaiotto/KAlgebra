"""
a1aodd_to_even_rgkalgebra.py
============================

`A1AoddToEvenRGKAlgebra(k)` — the odd Argyres–Douglas family
`A_𝖖([A_1, A_{2k+1}])` as an **`RGKAlgebra`**, the
odd analogue of the even family's flow.  A concrete RG
flow supplies only `auxiliary()` + `grading()` + `_s_rg_component()`
(and the identity `apex`); the whole `KAlgebra` API — `RG` (via the
exact per-charge solver `graded_rg_solver.solve_rg_exact`), `multiply`,
`rho`/`rho_inverse`, **and the trace** — is derived generically.  No slow
UV `BPS(A_{2k+1})` cluster graph is ever built (the operation this
construction exists to avoid).

The flow (drop the terminal node γ₁)
------------------------------------
`[A_1, A_{2k+1}]` is the single-node RG dropping the **terminal** node
`γ₁` of the linear `A_{2k+1}` BPS quiver `γ₁—γ₂—⋯—γ_{2k+1}`.  The
surviving chain `γ₂—⋯—γ_{2k+1} = A_{2k}` is the even theory
`[A_1, A_{2k}] = A1A2kKAlg(k)`.  The dropped `γ₁` is the **kernel** of
the degenerate `A_{2k+1}` pairing (`μ = γ₁+γ₃+⋯+γ_{2k+1}`, central), so
it adjoins as a spectator **U(1) flavour**:

    auxiliary  =  A1A2kKAlg(k).add_flavour(1)

— the even algebra flavour-rebased by the `KAlgebra.add_flavour` tool
(`flavoured_kalgebra.AddFlavourKAlgebra`), labels `(chord, (m,))` with
`m` the central μ-charge (= the γ₁-charge of the BPS state).  (Contrast
the even family's flow, where the dropped *pair* (γ₁,γ₂) is a
symplectic `QT(Z₂)` tensor factor; here the single dropped terminal is
central, so a flavour, not a torus.)

Grading (Γ_RG = Z = the μ / γ₁ charge)
--------------------------------------
The grading lattice is **Z** — the μ-power — with the obvious positive
cone `Z_{≥0}` and height `1`.  The survivor `A1A2kKAlg(k)` factor sits
entirely at grading-degree 0 (the "small" IR), so the height only sees
μ.  Because the survivor is degree-0, the UV canonical labels are the
auxiliary labels themselves `(chord, (m,))` and `apex` is the identity
(no charge↔chord identification needed); the genuine single-node flow
puts each survivor generator at the apex μ=0 and dresses it *upward*
(verified against `SubquiverRG(A_3, drop γ₁)`:
`RG((0,1,0)) = (0,1,0)+(1,1,0)`).

Spectrum generator
------------------
    S_RG  =  E_𝖖(X_{γ₁})  =  E_𝖖(μ · L)

— the (bare) terminal-node drop, with `L` a **short chord** of
`A1A2kKAlg(k)` (the g-element direction `L((1, H−2))`, `H = 2k+3`).
Component form (exact, finite, off-cone-vanishing):

    [S_RG]_{(m,)}  =  c_m · (L^m, (m,)),   c_m = (−q)^m / (q²;q²)_m,

a single auxiliary label per μ-power (`L^m = ((1, H−2, m),)`); the
degree-0 part is the identity `[S_RG]_0 = 1_B`.

Trace — μ-refined, generic exact-FS
-----------------------------------
The Schur index is the **μ-refined** (flavour-valued) inner product

    I_{a,b} = ⟨ RG_a·S_RG , RG_b·S_RG ⟩_aux
            = Σ_{c,d} [RG(a)·S_RG]_c · [RG(b)·S_RG]_d · I^aux_{c,d},

evaluated by the **generic engine's exact-FS bilinear pairing**
(`rgkalgebra.py`): it pairs *all* component pairs `(c,d)` through the
`add_flavour` auxiliary's `I^aux` (which keeps the μ-character), so the
index comes out flavour-valued in `R(U(1))` — e.g. for `[A_1, A_3]` the q²
flavour current is the SU(2) adjoint `χ_1 = μ + 1 + μ⁻¹` (the dropped
node's U(1) ⊂ the enhanced flavour SU(2)).  Because `_s_rg_component`
gives each graded piece exactly and the support walk is per-output-label,
the trace is truncation-safe to any `K` (no `L^{2·cutoff}` blow-up).
No hand-rolled μ-graded trace override is needed: the generic nested-aux
exact-FS engine reproduces the μ-refined index term-for-term, so the
class supplies only the flow data.
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
from a1a2k_kalg import A1A2kKAlg


def _e_q_coeff(m: int) -> HabiroElement:
    """`c_m = (−q)^m / (q²;q²)_m` — the m-th coefficient of `E_𝖖(X)` for a
    self-pairing-free generator `X` (⟨γ₁,γ₁⟩ = 0).  As a `HabiroElement`:
    numerator `(−1)^m q^m`, denominator `∏_{j=1}^{m}(1 − q^{2j})`."""
    num = LaurentPoly({m: (-1) ** m})
    denom = {j: 1 for j in range(1, m + 1)}
    return HabiroElement(num, denom)


class A1AoddToEvenRGKAlgebra(RGKAlgebra):
    """`[A_1, A_{2k+1}]` as a directional `RGKAlgebra` wrapping
    the flavour-rebased even algebra `A1A2kKAlg(k).add_flavour(1)`, with
    `RG` from the exact per-charge solver and the **generic exact-FS**
    μ-refined trace (no override).  See the module docstring."""

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._A = A1A2kKAlg(k)
        self._aux = self._A.add_flavour(1)      # μ = γ₁ (kernel) flavour
        self._H = self._A.H                      # = 2k + 3
        self._i0 = self._H - 2                    # short-chord ρ-index (g-element)

    # ----- RGKAlgebra contract -------------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = μ-power (label[1]); positive cone Z_{>=0}, height 1.
        return Grading(rank=1, deg=lambda lbl: tuple(lbl[1]), height=(1,),
                       cone_gens=((1,),))

    def apex(self, a):
        """Identity apex: UV labels coincide with auxiliary labels (the
        survivor is grading-degree 0)."""
        return (a[0], tuple(a[1]))

    def _short_chord_power(self, m: int):
        """`L^m` as an `A1A2kKAlg` label: the short-chord power
        `((1, H−2, m),)` (a single generator raised to `m`)."""
        return ((1, self._i0, m),)

    def _s_rg_component(self, p):
        """`[S_RG]_{(m,)}` — exact, finite, vanishing off the cone.

        `S_RG = E_𝖖(μ·L)` ⇒ degree-`m` part is the single dressed label
        `(L^m, (m,))` with coefficient `c_m`; degree 0 is the
        identity, negative degree empty."""
        (m,) = p
        if m < 0:
            return {}
        if m == 0:
            return {self._aux.identity(): _e_q_coeff(0)}
        label = (self._short_chord_power(m), (m,))
        return {label: _e_q_coeff(m)}

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG` as the q-order ≤ `cutoff` window (the fallback form): the
        μ-tower `{(L^m, (m,)): c_m}` up to `m < cutoff`.  The exact `RG` and
        trace paths use `_s_rg_component` instead."""
        out: dict = {}
        for m in range(cutoff):
            out.update(self._s_rg_component((m,)))
        return out

    # ----- flavour-aware section split -----------------------------------

    def _section_split(self, label):
        """The auxiliary's labels are `(chord, (m,))` — the flavour μ is the
        second coordinate (a tuple), not an elementwise charge on a flat
        vector, so the generic flat-vector `_section_split` (which does
        `label − section` componentwise) does not apply.  Disable the
        flavour-shift multiply cache here (return `flav = None`): `multiply`
        falls back to the direct `from_ir_image(RG(a)·RG(b))` per pair, which
        is correct (μ is central, so the product is well-defined either way)."""
        return tuple(label), None
