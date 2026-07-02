"""`U1A1AoddToEvenQTRGKAlgebra(k)` — the **second leg** of the A-type
`A1Aeven / U1A1Aodd` chain, as a **pure** `RGKAlgebra`:

    u(1)-gauged [A₁, A_{2k+1}]   ──drop a node──▶   [A₁, A_{2k}] ⊗ QT[Z²]

i.e. the UV is the gauged-odd theory `U1A1AoddKAlg(k)` and the IR is the
even AD theory one level down, `A1A2kKAlg(k)`, tensored with a rank-2
quantum torus (the ungauged U(1) gauge direction `f` paired with the
frozen flavour `v`, `B[f,v]=+1`).  This is the gauged→ungauged "ungauging"
flow used throughout the ConeKAlgebra explorations; the partner of the
**first leg** `A1AevenToU1AoddRGKAlgebra(k)`
(`[A₁,A_{2k+2}] → U1A1AoddKAlg(k)`).

Principled `S_RG`
-----------------
    S_RG  =  E_𝖖(X_{0,1} · L)

a **single** quantum-dilogarithm tower whose carrier is one **short chord
ray** `L` of `A1A2kKAlg(k)` tensored with the quantum-torus generator
`X_{0,1}` (the ungauged gauge direction).  Built **natively** — `carrier =
(L, X_{0,1})` q-commutes with itself (`⟨L,L⟩ = 0`, `X_{0,1}` self-commutes),
so `carrier^n` is a single auxiliary monomial and

    [S_RG]_{(n,)}  =  c_n · carrier^n,   c_n = (−q)^n / (q²;q²)_n,

with the auxiliary `TensorKAlgebra` cocycle absorbing all phases.  No
BPS-derived hard-coded `RG` tables are needed: here `RG` is **solved**
generically from the discovery relation
`RG(a)·S_RG = L_a + O(q)` over the cone-graded auxiliary — no spine, no
per-k extraction, uniform in k.

The short chord `L = ((1, i₀, 1),)` is taken with `i₀ = 0`; any short chord
gives a ρ-conjugate (hence isomorphic) flow, so the choice is a ρ-gauge.

Grading (Γ_RG = Z = the ungauged gauge / dropped-node charge)
------------------------------------------------------------
`Γ_RG = Z` is the `X_{0,1}` quantum-torus coordinate (the dropped node's
magnetic charge) — `deg((even_label, (c₁, c₂))) = c₂`, additive under the
auxiliary multiply (the even factor is QT-neutral); positive cone `Z_{≥0}`,
height 1.  `apex` is the identity (UV canonical labels are the auxiliary
cone labels; `RG` solved).

Validation
----------
The vacuum trace reproduces the gauged-odd Schur index **exactly** against
the standalone `U1A1AoddKAlg(k)` (k=1: `1 − q² + q¹⁰ + …`; the deep `q¹⁰`
term confirms the **exact-FS** bilinear trace captures structure a windowed
heuristic would truncate away), truncation-stable, on the generic engine
(no trace/multiply override).  The intrinsic axioms (bar / RG-multiplicative
/ orthonormality) pass.
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from habiro import HabiroElement
from rgkalgebra import RGKAlgebra
from grading import Grading
from tensor_kalgebra import TensorKAlgebra
from a1a2k_kalg import A1A2kKAlg
from quantum_torus_kalgebra import QuantumTorusKAlg

__all__ = ["U1A1AoddToEvenQTRGKAlgebra"]


def _e_q_coeff(n: int) -> HabiroElement:
    """`c_n = (−q)^n / (q²;q²)_n` — the n-th coefficient of `E_𝖖(X)` for a
    self-pairing-free carrier (`⟨carrier, carrier⟩ = 0`)."""
    return HabiroElement(LaurentPoly({n: (-1) ** n}),
                         {j: 1 for j in range(1, n + 1)})


class U1A1AoddToEvenQTRGKAlgebra(RGKAlgebra):
    """`u(1)-gauged [A₁, A_{2k+1}] → [A₁, A_{2k}] ⊗ QT[Z²]` as a pure generic
    `RGKAlgebra` with `S_RG = E_𝖖(X_{0,1}·L)` (short chord ray `L`).  See the
    module docstring."""

    def __init__(self, k: int, i0: int = 0):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._even = A1A2kKAlg(k)                      # IR even AD: [A₁, A_{2k}]
        self._qt = QuantumTorusKAlg([[0, 1], [-1, 0]])  # rank-2 torus (f, v)
        self._aux = TensorKAlgebra(self._even, self._qt)
        self._L = ((1, i0, 1),)                        # a short chord ray of the even
        self._carrier = (self._L, (0, 1))              # L · X_{0,1}
        # lazy tower cache: n -> (carrier^n label, c_n)
        self._tower: dict = {0: (self._aux.identity(), _e_q_coeff(0))}
        self._tower_cur = Element({self._aux.identity(): LaurentPoly.one()})
        self._carrier_el = Element({self._carrier: LaurentPoly.one()})

    # ----- lazy carrier-power tower --------------------------------------

    def _tower_term(self, n: int):
        """`(carrier^n label, c_n)` — `carrier` q-commutes with itself, so each
        power is a single auxiliary monomial; computed once and cached."""
        if n in self._tower:
            return self._tower[n]
        last = max(self._tower)
        cur = self._tower_cur
        for m in range(last + 1, n + 1):
            cur = self._aux.multiply_elements(cur, self._carrier_el)
            if len(cur.terms) != 1:
                raise RuntimeError(f"carrier^{m} is not a single monomial: {cur.terms}")
            self._tower[m] = (next(iter(cur.terms)), _e_q_coeff(m))
        self._tower_cur = cur
        return self._tower[n]

    # ----- RGKAlgebra contract -------------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = the X_{0,1} quantum-torus coordinate (dropped-node charge);
        # deg((even_label, (c1, c2))) = c2.  Positive cone Z_{>=0}, height 1.
        return Grading(rank=1, deg=lambda lbl: (lbl[1][1],),
                       height=(1,), cone_gens=((1,),))

    def _s_rg_component(self, p) -> dict:
        """`[S_RG]_{(n,)} = {carrier^n : c_n}` — a single auxiliary label per
        magnetic degree `n ≥ 0`; empty off the positive cone."""
        (n,) = p
        if n < 0:
            return {}
        lbl, c = self._tower_term(n)
        return {lbl: c}

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = E_𝖖(X_{0,1}·L)` windowed to magnetic degree ≤ `cutoff`."""
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        out: dict = {}
        for n in range(cutoff + 1):
            lbl, c = self._tower_term(n)
            out[lbl] = c
        return out

    # ----- flavour-aware section split (aux labels are not flat vectors) --

    def _section_split(self, label):
        return tuple(label), None

    def __repr__(self) -> str:
        return (f"U1A1AoddToEvenQTRGKAlgebra(k={self.k}) "
                f"[u(1)-gauged [A1,A{2*self.k+1}] → [A1,A{2*self.k}] ⊗ QT]")


if __name__ == "__main__":
    import warnings
    from u1a1aodd_kalg import U1A1AoddKAlg

    def ser(rps, K):
        return {e: str(r) for e, r in sorted(rps.coeffs.items())
                if e <= K and str(r) not in ("0", "")}

    for k in (1, 2):
        A = U1A1AoddToEvenQTRGKAlgebra(k)
        tgt = U1A1AoddKAlg(k)
        print(repr(A))
        print("  carrier S_RG = E_q(X_{0,1}·L), L =", A._L, " tower^1 =", A._tower_term(1)[0])
        print("  exact-FS available:", A._fs_exact_available())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            va = ser(A.trace(A.identity(), 12), 12)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            vt = ser(tgt.trace(tgt.identity(), 12), 12)
        print(f"  vacuum K12: mine={va}  target U1A1AoddKAlg({k})={vt}  match={va == vt}  warns={len(w)}")
