"""`SU2N2StarRGKAlgebra` / `U1GaugedSU2N2StarRGKAlgebra` — **N=2\* SU(2)**
(`T[SU(2), T*C³]`, the adjoint hyper) and its **U(1)-gauging**, as pure RG flows
to **pure SU(2)** by integrating out the adjoint matter with the **adjoint
quantum-dilog**

    S_RG  =  E_𝖖(μ·v²) · E_𝖖(μ) · E_𝖖(μ·v⁻²)     →  expand in χ_n(v),  χ_n → w_n

(the pure-SU(2) Wilson line `w_n`).  This is exactly `S = f_3(μ, w_n)`, the
identity `E_𝖖(μζ⁻¹)E_𝖖(μ)E_𝖖(μζ) = f_3(μ,χ_n(ζ))`: inserting the adjoint
factor into the pure-SU(2) index precisely converts the pure SU(2) calculation
to SU(2) N=2\*.  The **adjoint** counterpart of the fundamental-doublet matter
flow (`E(μv)E(μv⁻¹)`) and the gauged base rung `SU2Nf1PureSU2RGKAlgebra`.

Two realisations of the spectator `U(1)` (the central `μ = X_{½,½,½}`):

* **`SU2N2StarRGKAlgebra`** — `μ` a **flavour**: aux = `PureSU2KAlg() ⊗
  QuantumTorusKAlg([[0]])` (the central `U(1)_F` torus, `Tr(vᵐ)=μᵐ`); graded by
  the μ-charge.  `Tr(1)` reproduces the N=2\* Schur index (the contour
  formula `I(q;μ)`) **exactly through q⁹** — even q-powers identical, odd
  q-powers identical up to `μ → −μ` (the `E_𝖖(z)=E^{paper}(−z)` sign), i.e.
  `Tr(1)(μ) = I(q; −μ)`.

* **`U1GaugedSU2N2StarRGKAlgebra`** — `μ` **weakly gauged** (user, 2026-06-27;
  paper §"Improved intertwining recursion"): aux = `PureSU2KAlg() ⊗
  QuantumTorusKAlg([[0,1],[-1,0]])` (the `U(1)` *gauge* torus), the adjoint on
  the electric gauge leg `X_{0,1}`, graded by `b`.  Gauging the Cartan of the
  flavour enlarges Γ = Z³ → Z⁴ and makes the pairing **non-degenerate** — curing
  the famous non-normalizability of the bare N=2\* spherical vector; the extended
  algebra contains both N=2\* and **SQED₃** as sub-algebras.  Flavourless
  `Tr(1) = 1 − q² + q⁶ + 4q⁸ + 4q¹⁰ + …` = `(q²;q²)_∞² · μ⁰[I(q;μ)]` (the
  `U(1)`-Haar of the N=2\* index), truncation-stable.

The adjoint `S_RG` (shared)
---------------------------
At μ-charge `k` the adjoint factor `Σ_{a+b+d=k} c_a c_b c_d v^{2(a-d)}`
(`c_m = E_𝖖`-coeff) is even-palindromic in `v`, so it decomposes into the
**even** Wilson characters (`χ₂²=χ₄+χ₂+χ₀, …`):

    [S_RG]_{(k,)} = Σ_{n even, 0 ≤ n ≤ 2k}  N_{k,n} · (w_n, X-leg^k),
    N_{k,n} = P_k[v^n] − P_k[v^{n+2}],   P_k[v^j] = Σ_{a−d = j/2, a+b+d = k} c_a c_b c_d.

Both flows are **pure** `RGKAlgebra`s (generic exact-FS engine, no override),
spine-free.  Validation: `tests/test_su2_n2star_rgkalgebra.py`.
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

__all__ = ["SU2N2StarRGKAlgebra", "U1GaugedSU2N2StarRGKAlgebra"]


def _w(n: int):
    """The pure-SU(2) Wilson line `w_n` (= `χ_n`): identity `()` for `n = 0`."""
    if n == 0:
        return ()
    return ((("W", n), 1),)


def _c(m: int):
    """The `E_𝖖` coefficient `c_m` (Habiro-exact), or `None` for `m < 0`."""
    if m < 0:
        return None
    return eq_coeff(m)


def _Pk(k: int, j: int):
    """`P_k[v^j] = Σ_{a−d=j/2, a+b+d=k, a,b,d≥0} c_a c_b c_d` (the `v^j` coeff of
    the adjoint factor `E(μv²)E(μ)E(μv⁻²)` at μ-charge `k`); `None` if `j` odd or
    the sum is empty."""
    if j % 2:
        return None
    s = j // 2
    acc = None
    a = max(0, s)
    while True:
        d = a - s
        if d < 0:
            a += 1
            continue
        b = k - a - d
        if b < 0:
            break
        term = _c(a) * _c(d) * _c(b)
        acc = term if acc is None else (acc + term)
        a += 1
    return acc


def _adjoint_s_rg_component(aux, leg, k: int) -> dict:
    """`[S_RG]_{(k,)}` for the adjoint matter, `leg(k)` placing the μ-charge `k`
    on the chosen torus leg.  `Σ_{n even} N_{k,n} (w_n, leg(k))`,
    `N_{k,n} = P_k[v^n] − P_k[v^{n+2}]`."""
    if k < 0:
        return {}
    if k == 0:
        return {aux.identity(): eq_coeff(0)}
    out: dict = {}
    for n in range(2 * k, -1, -2):
        top = _Pk(k, n)
        nxt = _Pk(k, n + 2)
        coeff = top if nxt is None else (top - nxt)
        if coeff is not None and not coeff.is_zero():
            out[(_w(n), leg(k))] = coeff
    return out


class _AdjointMatterRGKAlgebra(RGKAlgebra):
    """Shared engine for N=2\* (adjoint matter) over `pure SU(2) ⊗ (μ-torus)`.
    Subclasses fix the μ-torus, the leg placement, and the grading."""

    _MU_TORUS: list
    _GRADE_INDEX: int          # which lbl[1] coordinate carries the μ-charge

    def __init__(self) -> None:
        self._su2 = PureSU2KAlg()
        self._mu = QuantumTorusKAlg(self._MU_TORUS)
        self._aux = TensorKAlgebra(self._su2, self._mu)

    def auxiliary(self):
        return self._aux

    def _leg(self, k: int):
        raise NotImplementedError

    def grading(self) -> Grading:
        gi = self._GRADE_INDEX
        return Grading(rank=1, deg=lambda lbl: (lbl[1][gi],),
                       height=(1,), cone_gens=((1,),))

    def _s_rg_component(self, p) -> dict:
        (k,) = p
        return _adjoint_s_rg_component(self._aux, self._leg, k)

    def rg_generator(self, cutoff: int) -> dict:
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        out: dict = {}
        for k in range(cutoff + 1):
            out.update(self._s_rg_component((k,)))
        return out

    def _section_split(self, label):
        return tuple(label), None


class SU2N2StarRGKAlgebra(_AdjointMatterRGKAlgebra):
    """N=2\* SU(2) (adjoint matter, **μ a flavour**) → pure SU(2).  `aux = pure
    SU(2) ⊗ U(1)_F flavour torus`; `Tr(1) = I(q; −μ)` (the N=2\* Schur
    index).  See the module docstring."""

    _MU_TORUS = [[0]]              # central U(1)_F flavour torus
    _GRADE_INDEX = 0              # μ-charge is the rank-1 torus coordinate

    def _leg(self, k: int):
        return (k,)

    def __repr__(self) -> str:
        return "SU2N2StarRGKAlgebra(N=2* SU(2), μ-flavour → pure SU(2))"


class U1GaugedSU2N2StarRGKAlgebra(_AdjointMatterRGKAlgebra):
    """U(1)-gauged N=2\* SU(2) (**μ weakly gauged**) → pure SU(2) ⊗ QT[Z²].  The
    adjoint on the electric gauge leg `X_{0,1}`; Γ = Z⁴ non-degenerate
    (normalizable), SQED₃ a sub-algebra.  Flavourless `Tr(1) = 1 − q² + q⁶ + 4q⁸
    + 4q¹⁰ + …`.  See the module docstring."""

    _MU_TORUS = [[0, 1], [-1, 0]]    # U(1) gauge torus QT[Z²]
    _GRADE_INDEX = 1                 # gauge-leg charge b = lbl[1][1]

    def _leg(self, k: int):
        return (0, k)                # X_{0,k} on the electric gauge leg

    def __repr__(self) -> str:
        return "U1GaugedSU2N2StarRGKAlgebra(U(1)-gauged N=2* SU(2) → pure SU(2) ⊗ QT[Z²])"


if __name__ == "__main__":
    import warnings
    for cls in (SU2N2StarRGKAlgebra, U1GaugedSU2N2StarRGKAlgebra):
        T = cls()
        print(repr(T), " coeff =", T.coefficient_ring(), " fs_exact =", T._fs_exact_available())
        print("  S_RG levels 0..2 (adjoint, even χ):")
        for k in range(3):
            print("    k=%d:" % k, {l: str(c) for l, c in T._s_rg_component((k,)).items()})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vac = T.trace(T.identity(), 8)
            print("  Tr(1):", {e: str(r) for e, r in sorted(vac.coeffs.items())}, " warns =", len(w))
        print()
