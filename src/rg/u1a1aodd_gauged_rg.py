"""
u1a1aodd_gauged_rg.py
=====================

`U1A1AoddGaugedRG(k)` — the **gauged** odd Argyres-Douglas family,
u(1)-gauged `A_𝖖([A_1, A_{2k+1}])`, as a new-contract `RGKAlgebra`
.  The gauged sibling of the ungauged
`a1aodd_to_even_rgkalgebra.A1AoddToEvenRGKAlgebra`.

Same RG flow as the ungauged oracle — drop the terminal node γ₁ of the
linear `A_{2k+1}` BPS quiver, `S_RG = E_𝖖(X_e · L)` — but the auxiliary
is the **unflavoured** tensor

    auxiliary = A1A2kKAlg(k) ⊗ QT(Z²)

instead of the U(1)-flavoured `A1A2kKAlg(k).add_flavour(1)` (the author's
prescription).  Gauging the U(1) promotes the spectator flavour μ to one
leg (electric `X_e`) of a symplectic quantum torus `QT(Z²)`; the magnetic
leg's non-commutativity `X_e X_m = q X_m X_e` is what balances each short
chord into an honest single-chord ray (no double-chords) — see
the design notes and the design notes.

Geometry (triangle slice).  The target lives on the (2k+4)-gon; the
auxiliary `A1A2kKAlg(k)` on the (2k+3)-gon; they differ by slicing off a
triangle along a length-2 chord (the dressing chord `L`).  Chords that
miss the triangle map directly to `A1A2k` chords (undressed RG); chords
crossing it pick up the `E_𝖖(X_e·L)` dressing.

Labels:  `(chord, (c0, c1))` — `chord` an `A1A2kKAlg(k)` label, `(c0, c1)`
the QT(Z²) charge: `c1` the **dressed** leg (0,1) = the old ungauged μ
(Γ_RG = Z grades by `c1`), `c0` the **free gauge** leg (1,0).

Validation (k=1, the suite in the source repository).  CONFIRMED
`U1A1AoddGaugedRG(1) ≅ BPSKAlgebra(B_GAUGED)` = the genuine gauged
hexagon = `U1HexagonKAlg`'s ground truth, via the explicit
canonical-basis bijection `φ = A · ocharge` where `A`'s columns (images
of the oracle charge basis e1,e2=pentagon A_2 dirs, e3=gauge (1,0),
e4=dressed (0,1)) are `(0,1,0,0), (0,0,1,0), (0,-1,0,-1), (1,0,1,0)`.
`A` intertwines the pairings exactly (`Aᵀ B_GAUGED A = B_oracle`), so the
cocycle q-powers match — exact structure-constant agreement on all
sampled products.  Also bar-invariant, associative, orthonormal.
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
from tensor_kalgebra import TensorKAlgebra
from quantum_torus_kalgebra import QuantumTorusKAlg
from a1a2k_kalg import A1A2kKAlg


def _e_q_coeff(m: int) -> HabiroElement:
    """`c_m = (−q)^m / (q²;q²)_m` — the m-th coefficient of `E_𝖖(X)` for a
    self-pairing-free generator (⟨e,e⟩ = 0).  Habiro form: numerator
    `(−1)^m q^m`, denominator `∏_{j=1}^{m}(1 − q^{2j})`."""
    num = LaurentPoly({m: (-1) ** m})
    denom = {j: 1 for j in range(1, m + 1)}
    return HabiroElement(num, denom)


class U1A1AoddGaugedRG(RGKAlgebra):
    """`u(1)-gauged [A_1, A_{2k+1}]` as an `RGKAlgebra` over
    `A1A2kKAlg(k) ⊗ QT(Z²)`, dropping the terminal node γ₁ with
    `S_RG = E_𝖖(X_e · L)` (electric leg dressed by the short chord `L`)."""

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._surv = A1A2kKAlg(k)
        self._qt = QuantumTorusKAlg([[0, 1], [-1, 0]])
        self._aux = TensorKAlgebra(self._surv, self._qt)
        self._H = self._surv.H        # = 2k + 3
        self._i0 = self._H - 2        # short dressing chord (g-element)

    # ----- RGKAlgebra contract -------------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = the DRESSED QT leg (0,1) charge (= the old ungauged μ;
        # user: "same as ungauged, replace μ with X_(0,1)").  The other QT
        # leg (1,0) is the free gauge direction.  Cone Z_{>=0}, height 1.
        return Grading(rank=1, deg=lambda lbl: (lbl[1][1],), height=(1,),
                       cone_gens=((1,),))

    def apex(self, a):
        """Identity apex: UV labels coincide with auxiliary labels."""
        return (a[0], tuple(a[1]))

    def _short_chord_power(self, m: int):
        """`L^m` as an `A1A2kKAlg` label: the short dressing chord to
        power `m` (a single generator `((1, H−2, m),)`)."""
        return ((1, self._i0, m),)

    def _s_rg_component(self, p):
        """`[S_RG]_{(m,)}` — exact, finite, vanishing off the cone.

        `S_RG = E_𝖖(X_{(0,1)} · L)` ⇒ degree-`m` part (in the dressed
        leg (0,1)) is the single label `(L^m, (0, m))` with Habiro
        coefficient `c_m`; degree 0 is the auxiliary identity, negative
        degree empty.  The gauge leg (1,0) charge stays 0 in S_RG."""
        (m,) = p
        if m < 0:
            return {}
        if m == 0:
            return {self._aux.identity(): _e_q_coeff(0)}
        label = (self._short_chord_power(m), (0, m))
        return {label: _e_q_coeff(m)}

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG` windowed to q-order ≤ `cutoff`: the electric μ-tower
        `{(L^m, (m, 0)): c_m}` for `m < cutoff`."""
        out: dict = {}
        for m in range(cutoff):
            out.update(self._s_rg_component((m,)))
        return out

    # ----- flavour-aware section split -----------------------------------

    def _section_split(self, label):
        """Disable the flavour-shift multiply cache (the QT charge is a
        2-tuple, not a flat-vector charge): `multiply` falls back to the
        direct `from_ir_image(RG(a)·RG(b))` per pair."""
        return tuple(label), None

    # ----- adaptive-window trace (certified FS object) -------------------

    def trace(self, a, K: int = 20):
        """`Tr_UV(L_a) = Tr_aux(ρ(S_RG)·RG(a)·S_RG)` with a **multi-window
        stable `S_RG` q-order window** (scaled to the linear Habiro valuation).

        The generic `RGKAlgebra.trace` windows `S_RG` at the fixed
        `_rg_cutoff()` for *every* `a`; q-order is non-additive across
        `RG(a)·S_RG` (the quantum-torus pairing phase `q^{⟨γ,γ'⟩}` can be
        negative), so for a high-charge `a` that fixed window leaves the
        FS object — hence the trace — under-resolved (`rg_times_s_rg`'s
        documented caveat).

        Sizing the window.  The Habiro coefficient `c_m = (−q)^m/(q²;q²)_m`
        (`_e_q_coeff`) has q-valuation **m** — LINEAR in `m` — so an `S_RG`
        term enters at q-order `m`, and after the (bounded, linear-in-`m`)
        pairing shift it can reach `q^{≤K}` only for `m = O(K)`.  The window
        must therefore grow as `O(K)`; the old ceiling `2K + cutoff + 6`
        could fall short, and — worse — the old criterion stopped at the
        **first** pair of consecutive (`+2`) windows that agreed on `q^{≤K}`.
        A truncation plateau that merely persists across two windows fooled
        that test into stopping early (the documented "gauged-RG oracle is
        unreliable beyond ~q²⁹ at k=3" artifact: spurious / growing high-q
        terms).  We instead require the `q^{≤K}` series to be stable across
        **several** consecutive windows before accepting, under a generous
        `win_max = 6K + 48` ceiling — so a transient plateau cannot certify a
        premature, under-resolved result."""
        cache = self.__dict__.setdefault("_adaptive_trace_cache", {})
        ck = (a, K)
        if ck in cache:
            return cache[ck]
        aux = self.auxiliary()
        rg_a = self.RG(a)
        prev = None
        result = None
        agree_run = 0
        win = max(self._rg_cutoff(), 2)
        win_max = 6 * K + 48
        while win <= win_max:
            try:
                s_rg = self._s_rg_as_aux_element(win, K + win)
                rho_s = aux.rho_element(s_rg)
                a_s = aux.multiply_elements(rg_a, s_rg)
                prod = aux.multiply_elements(rho_s, a_s)
                tr = aux.trace_element(prod, K)
            except RuntimeError:
                break              # aux cannot reduce this window's powers
            if prev is not None and tr.coeffs == prev.coeffs:
                agree_run += 1
                if agree_run >= 3:   # stable across >=3 consecutive windows
                    result = tr      # (a transient +2 plateau cannot survive)
                    break
            else:
                agree_run = 0
            prev = tr
            win += 2
        if result is None:
            result = prev
        cache[ck] = result
        return result


if __name__ == "__main__":
    T = U1A1AoddGaugedRG(1)
    A = T._surv
    print("k=1: aux =", type(T.auxiliary()).__name__,
          "(", type(A).__name__, "⊗", type(T._qt).__name__, ")",
          " H =", T._H, " dressing i0 =", T._i0)
    print("identity =", T.identity())
    # RG of a survivor generator at electric/magnetic 0.
    g = (((1, 0, 1),), (0, 0))
    print("RG(g(1,0)@(0,0)) =", {l: str(c) for l, c in T.RG(g).terms.items()})
    # A couple of products.
    h = (((1, 1, 1),), (0, 0))
    print("multiply(g(1,0)@(0,0), g(1,1)@(0,0)) =",
          {l: str(c) for l, c in T.multiply(g, h).terms.items()})
