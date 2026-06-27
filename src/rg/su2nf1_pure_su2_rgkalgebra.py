"""`SU2Nf1PureSU2RGKAlgebra` — **U(1)-gauged SU(2) N_f=1** as the RG flow **to
`pure SU(2) ⊗ QT[Z²]`**, integrating out the single **gauge-SU(2)-doublet**
matter hyper `S_RG = E_𝖖(X_{0,1}·v)·E_𝖖(X_{0,1}·v⁻¹)`.

The **base matter rung of the SU(2)-gauged chain** — the SU(2)-gauge analog of
`A1D3Sqed2RGKAlgebra` (#675) / the even-rung `U1A1DevenSqedRGKAlgebra`, gauging
the flavour SU(2) of the A1Dodd–U1A1Deven chain throughout (base **torus → pure
SU(2)**; user, 2026-06-27):

    SU(2)-flavoured  :  torus    ← SQED₂(=A1D2)      ← A1D3        ← U1A1D4            ← …
    SU(2)-gauged     :  pureSU2  ← U(1)-gauged-SU2Nf1 ← SU(2)·A1D3 ← SU(2)×U(1)·A1D4    ← …
                                  ↑ THIS  (the matter rung;  → pure SU(2) × QT[Z²])

The gauged doublet `S_RG` — `E(X₀₁ v) E(X₀₁/v)`, χ-expanded, `χ_n → w_n`
---------------------------------------------------------------------------
In the SU(2)-**flavoured** chain the doublet folds into a *single* `E_𝖖(X·L)`
(the chord `L` carrying both flavour weights `v^{±1}` as a coefficient).  Here
the SU(2) is **gauged** — a *dynamical* pure-SU(2) factor — so the matter quark
is the honest two-weight gauge doublet on the **electric gauge leg** `X_{0,1}`
(user's recipe, 2026-06-27):

    expand   E_𝖖(X_{0,1} v) · E_𝖖(X_{0,1} v⁻¹)   in SU(2) characters  χ_n(v),
    then replace each  χ_n(v)  by the pure-SU(2) **Wilson line** `w_n`.

Grouping by the gauge charge `N` (the `X_{0,1}` power) and the telescoping
χ-peel of the palindromic `Σ_{a+b=N} c_a c_b v^{a-b}`:

    [S_RG]_{(N,)}  =  Σ_{n ≡ N (2), 0 ≤ n ≤ N}  d_n(N) · (w_n , X_{(0,N)})
    d_n(N)  =  c_{(N+n)/2} c_{(N-n)/2}  −  c_{(N+n)/2+1} c_{(N-n)/2-1}     (c_m = E_𝖖-coeff, 0 for m<0)

e.g. `[S_RG]_2 = c_2·w_2 + (c_1²−c_2)·w_0` on `X_{(0,2)}` (Habiro-exact via
`sunf_dilog.eq_coeff`).

Defining data (a **pure** `RGKAlgebra` — generic exact-FS engine, no override)
-----------------------------------------------------------------------------
* `auxiliary()` = `pure SU(2) (cone) ⊗ QT[Z²]` = `PureSU2KAlg() ⊗
  QuantumTorusKAlg([[0,1],[-1,0]])`.  Spine-free, flavourless (`TrivialZPlusRing`
  — the SU(2) is gauged).  Labels `(su2_label, (a, b))`: `su2_label` a pure-SU(2)
  cone label (Wilson `w_n`), `(a, b)` the gauge QT charge.
* `grading()` = `Γ_RG = Z` = the **gauge leg charge `b`** — the
  `X_{ab}·L_c ↦ b` read-off (`deg(lbl)=lbl[1][1]`); height 1, positive cone
  `Z_{≥0}`.  (The *even*-rung gauge-grading pattern, as in `U1A1Deven`.)
* `apex` = identity (UV canonical labels are the auxiliary labels).

Validation: `Tr(1)` reproduces the
documented U(1)-gauged SU(2) N_f=1 Schur index `1 − q² − q⁴ − q⁶ + q⁸ + 2q¹⁰`,
truncation-stable, `w₁² = 1 + w₂` (gauge Wilson
Clebsch–Gordan), orthonormal, spine-free.  It **agrees with `UNNfKAlgebra(2,1)`
(U(2) N_f=1) through q¹⁰** and differs at **q¹²** (0 vs 2) — the genuine
**SU(2) N_f=1 ⊂ U(2) N_f=1** subalgebra / global-form difference (user,
2026-06-27: "almost the same … one a subalgebra of the other"), *not* a
truncation artifact (stable at K=12/14/16).
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
from pure_su2_h_cone_data import PureSU2KAlg
from sunf_dilog import eq_coeff

__all__ = ["SU2Nf1PureSU2RGKAlgebra"]


def _w(n: int):
    """The pure-SU(2) Wilson line `w_n` (= the spin-`n/2` character `χ_n`, cone
    label `(0,n)` in `(m,e)`): the identity `()` for `n = 0`, else the single
    Wilson generator `('W', n)` to exponent 1."""
    if n == 0:
        return ()
    return ((("W", n), 1),)


def _c(m: int):
    """The `E_𝖖` coefficient `c_m` (Habiro-exact), or `None` for `m < 0`
    (a missing dilog factor — the convention `c_{<0} = 0`)."""
    if m < 0:
        return None
    return eq_coeff(m)


class SU2Nf1PureSU2RGKAlgebra(RGKAlgebra):
    """U(1)-gauged SU(2) N_f=1 as the pure RG flow
    `S_RG = E_𝖖(X₀₁v)E_𝖖(X₀₁v⁻¹)` (a two-weight gauge doublet, χ-expanded with
    `χ_n → w_n`) to `pure SU(2) ⊗ QT[Z²]`.  The base matter rung of the
    SU(2)-gauged chain (→ pure SU(2)).  See the module docstring."""

    def __init__(self) -> None:
        self._su2 = PureSU2KAlg()                          # pure SU(2), cone (exact trace)
        self._qt = QuantumTorusKAlg([[0, 1], [-1, 0]])     # the gauge torus QT[Z²]
        self._aux = TensorKAlgebra(self._su2, self._qt)

    # ----- RGKAlgebra defining data --------------------------------------

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = the gauge leg charge b (X_{ab}·L_c ↦ b) = lbl[1][1]; cone Z_{>=0}.
        return Grading(rank=1, deg=lambda lbl: (lbl[1][1],),
                       height=(1,), cone_gens=((1,),))

    def _s_rg_component(self, p) -> dict:
        """`[S_RG]_{(N,)}` — the gauge doublet `E_𝖖(X₀₁v)E_𝖖(X₀₁v⁻¹)` at gauge
        charge `N`, expanded in characters and `χ_n → w_n`:

            Σ_{n ≡ N (2), 0 ≤ n ≤ N}  d_n(N) · (w_n, X_{(0,N)}),
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
            term = _c(a) * _c(b)                          # both ≥ 0 here
            lo, hi = _c(a + 1), _c(b - 1)
            if lo is not None and hi is not None:
                term = term - lo * hi
            if not term.is_zero():
                out[(_w(n), (0, N))] = term
        return out

    def rg_generator(self, cutoff: int) -> dict:
        """`S_RG = E_𝖖(X₀₁v)E_𝖖(X₀₁v⁻¹)` (the χ-expanded doublet tower) windowed
        to gauge charge `≤ cutoff`."""
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
        return ("SU2Nf1PureSU2RGKAlgebra("
                "U(1)-gauged SU(2) N_f=1 → pure SU(2) ⊗ QT[Z²])")


if __name__ == "__main__":
    import warnings
    T = SU2Nf1PureSU2RGKAlgebra()
    print(repr(T), " coeff =", T.coefficient_ring())
    print("  aux =", type(T.auxiliary()).__name__, " fs_exact =", T._fs_exact_available())
    print("  S_RG levels 0..3 (gauge doublet E(X₀₁v)E(X₀₁/v), χ_n→w_n):")
    for N in range(4):
        print("    N=%d:" % N, {l: str(c) for l, c in T._s_rg_component((N,)).items()})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vac = T.trace(T.identity(), 10)
        print("  vacuum index (= U(1)-gauged SU(2) N_f=1):",
              {e: str(r) for e, r in sorted(vac.coeffs.items())}, " warns =", len(w))
        wsq = T.multiply((_w(1), (0, 0)), (_w(1), (0, 0)))
        print("  w₁² =", {l: str(c) for l, c in wsq.terms.items()})
