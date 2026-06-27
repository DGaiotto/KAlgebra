"""
a1deven_rgkalgebra.py
=====================

`A1DevenRGKAlgebra(k)` — the **even** D-type Argyres–Douglas family
`A_𝖖([A_1, D_{2k+2}])` as a new-contract `RGKAlgebra` (Plan 20), the
two-fork analogue of `A1AoddToEvenRGKAlgebra` (the single-drop odd
prototype).  A concrete RG flow supplies only `auxiliary()` +
`grading()` + `_s_rg_component()` (and the identity `apex`); the whole
`KAlgebra` API — `RG`, `multiply`, `rho`/`rho_inverse`, `trace` — is
derived generically.  No slow UV `BPS(D_{2k+2})` cluster graph is built.

The flow (drop the two D-fork terminals)
----------------------------------------
The `D_{2k+2}` Dynkin/BPS quiver is the chain `γ_1—⋯—γ_{2k}` with **two**
extra terminal nodes `γ', γ''` both attached to the chain end `γ_{2k}`
(the trivalent fork).  Dropping the two fork terminals leaves the chain
`A_{2k}` — the even theory `[A_1, A_{2k}] = A1A2kKAlg(k)`.

Both dropped terminals attach to the *same* survivor node, so both are
kernels of the degenerate pairing and adjoin as a **U(2)** spectator
flavour (their two central charges `μ_1, μ_2` are the U(2) Cartan, and
the exchange `γ' ↔ γ''` enhances `U(1)²` to `U(2)`):

    auxiliary  =  A1A2kKAlg(k).add_flavour(SU2xU1ZPlusRing())

— labels `(chord, (κ, m))` with `κ` the SU(2) χ-index (spin κ/2) and `m`
the U(1) charge (= total μ-power).  (Contrast the odd
`A1AoddToEvenRGKAlgebra`, where the *single* dropped terminal is one
central U(1) flavour.)

Spectrum generator (two collinear quantum dilogs → U(2) characters)
-------------------------------------------------------------------
    S_RG  =  E_𝖖(μ_1 · L) · E_𝖖(μ_2 · L)

— both fork hypers couple to the **same** short chord `L = L((1, H−2))`
of `A1A2kKAlg(k)` (`H = 2k+3`), so the two `E_𝖖`'s are collinear (commute)
and combine into the U(2) character generating function.  Expanding in
U(2) = SU(2)×U(1) irreps, the total level `N` (U(1) charge) part is a
finite sum over the GL(2) Young diagrams `(a, b)`, `a ≥ b ≥ 0`, `a+b = N`:

    [S_RG]_{(N,)}  =  Σ_{a+b=N, a≥b}  (c_a c_b − c_{a+1} c_{b-1})
                          · ( L^N , (κ = a−b, m = N) ),
    c_m = (−q)^m / (q²;q²)_m,

the Schur coefficient `c_a c_b − c_{a+1} c_{b-1}` being the 2-variable
inverse-Kostka (Weyl) peel of `Σ c_{m_1} c_{m_2} μ_1^{m_1} μ_2^{m_2}`.
Degree 0 is the identity (`(a,b)=(0,0)`, the U(2) singlet).

Grading (Γ_RG = Z = the U(1) charge / total μ-power)
----------------------------------------------------
`Γ_RG = Z` is the U(1) flavour charge `m` (the total number of fork
hypers), positive cone `Z_{≥0}`, height 1.  The survivor `A1A2kKAlg(k)`
sits at degree 0, so the UV labels coincide with the auxiliary labels and
`apex` is the identity; the SU(2) χ-structure lives *within* each grade as
the flavour character (the U(2) doublet's spin content).

Status
------
Even-D companion to the odd `A1AoddToEvenRGKAlgebra`; together they are
the `a1_{A/D} → a1_even` matter-dressing programme over `A1A2kKAlg`.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from habiro import HabiroElement
from rgkalgebra import RGKAlgebra
from grading import Grading
from zplus_ring import SU2xU1ZPlusRing
from a1a2k_kalg import A1A2kKAlg
from a1aodd_to_even_rgkalgebra import _e_q_coeff       # c_m = (−q)^m/(q²;q²)_m


def _u2_char_coeff(a: int, b: int) -> HabiroElement:
    """The U(2) Schur coefficient `c_a c_b − c_{a+1} c_{b-1}` (`a ≥ b ≥ 0`) of
    the irrep `s_{(a,b)}` in `E_𝖖(μ_1 L) E_𝖖(μ_2 L) = Σ c_{m_1} c_{m_2}
    μ_1^{m_1} μ_2^{m_2} L^{m_1+m_2}` — the 2-variable inverse-Kostka peel
    (`c_{-1} = 0`)."""
    term = _e_q_coeff(a) * _e_q_coeff(b)
    if b >= 1:
        term = term - _e_q_coeff(a + 1) * _e_q_coeff(b - 1)
    return term


class A1DevenRGKAlgebra(RGKAlgebra):
    """`[A_1, D_{2k+2}]` as a directional new-contract `RGKAlgebra` wrapping the
    U(2)-flavoured even algebra `A1A2kKAlg(k).add_flavour(SU2xU1ZPlusRing())`,
    with `S_RG = E_𝖖(μ_1 L) E_𝖖(μ_2 L)`.  See the module docstring."""

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._A = A1A2kKAlg(k)
        self._aux = self._A.add_flavour(SU2xU1ZPlusRing())   # U(2) = SU(2)×U(1)
        self._H = self._A.H                                   # = 2k + 3
        self._i0 = self._H - 2                                # short-chord ρ-index

    # ----- RGKAlgebra contract -------------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = U(1) charge (label[1][1]); positive cone Z_{>=0}, height 1.
        return Grading(rank=1, deg=lambda lbl: (lbl[1][1],), height=(1,),
                       cone_gens=((1,),))

    def apex(self, a):
        """Identity apex: UV labels coincide with auxiliary labels (the survivor
        is grading-degree 0)."""
        return (a[0], tuple(a[1]))

    def _short_chord_power(self, N: int):
        """`L^N` as an `A1A2kKAlg` chord label: `((1, H−2, N),)` (the identity
        `()` for `N = 0`)."""
        if N == 0:
            return ()
        return ((1, self._i0, N),)

    def _s_rg_component(self, p):
        """`[S_RG]_{(N,)}` — exact, finite, vanishing off the cone.

        `S_RG = E_𝖖(μ_1 L) E_𝖖(μ_2 L)` ⇒ the total-level-`N` part is the sum
        over U(2) irreps `(a, b)` (`a+b=N`, `a≥b≥0`) of the chord `L^N` carried
        at flavour `(κ = a−b, m = N)` with Schur coefficient
        `c_a c_b − c_{a+1} c_{b-1}`; degree 0 is the U(2) singlet (identity)."""
        (N,) = p
        if N < 0:
            return {}
        chord = self._short_chord_power(N)
        out: dict = {}
        for kappa in range(N, -1, -2):           # κ = a − b ∈ {N, N−2, …, 0/1}
            a = (N + kappa) // 2
            b = (N - kappa) // 2
            coeff = _u2_char_coeff(a, b)
            if coeff.is_zero():
                continue
            out[(chord, (kappa, N))] = coeff
        return out

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG` as the level ≤ `cutoff` window (the fallback / `trace` form):
        the U(2)-character tower `{(L^N, (κ, N)): coeff}` for `N < cutoff`."""
        out: dict = {}
        for N in range(cutoff):
            out.update(self._s_rg_component((N,)))
        return out

    # ----- trace: the generic exact-FS U(2)-refined Schur index (no override) -
    # The hand-rolled U(1)-graded rg_s_graded/inner_product/trace that lived here
    # predated the nested-aux exact-FS engine (#666); the generic engine's
    # bilinear exact-FS pairing now reproduces the U(2)-refined index
    # term-for-term and faster (it pairs all components through the
    # add_flavour(SU2×U1) aux's I^aux, keeping the full U(2) character — e.g. the
    # U(2) adjoint flavour current `1 + χ_{(1,±1)} + χ_{(2,0)}` at q² of
    # [A_1, D_4]).  So `trace`/`inner_product` are inherited from RGKAlgebra.

    # ----- flavour-aware section split -----------------------------------

    def _section_split(self, label):
        """The auxiliary labels are `(chord, (κ, m))` — the flavour is a tuple,
        not a flat-vector charge — so disable the flavour-shift multiply cache
        (return `flav = None`): `multiply` falls back to the direct
        `from_ir_image(RG(a)·RG(b))` per pair (correct: the flavour is
        central)."""
        return tuple(label), None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for k in (1, 2):
        T = A1DevenRGKAlgebra(k)
        H = T._H
        print(f"\n{'='*70}\n  A1DevenRGKAlgebra(k={k})  =  [A_1, D_{2*k+2}]"
              f"   (survivor A_{2*k}, H={H})\n{'='*70}")
        print("  S_RG = E_q(mu1 L) E_q(mu2 L), expanded in U(2)=SU(2)xU(1) chars:")
        for N in range(3):
            comp = T._s_rg_component((N,))
            print(f"    level N={N}:")
            for (chord, (kappa, m)), c in sorted(comp.items(), key=lambda t: -t[0][1][0]):
                print(f"      s_(a,b) chi={kappa} U(1)={m}  chord={chord}:  {c}")
        # RG of a survivor generator (short chord L itself).
        L = (T._A.L((1, 0)), (0, 0))     # survivor gen, flavour-trivial
        rg = T.RG(L)
        print(f"  RG(L((1,0))) has {len(rg.terms)} terms:")
        for lab, c in sorted(rg.terms.items(), key=lambda t: t[0][1][1]):
            print(f"      {lab}:  {c}")
