"""**"Wild" `RGKAlgebra`s** — formal RG flows whose `S_RG = E_𝖖(L_a)` sits on an
arbitrary **monomial cone ray** `L_a`, chosen for the *fun of it* rather than to
realise a known 4d N=2 theory.  They stress the machinery: the generic exact-FS
engine still produces a well-defined, truncation-safe algebra (closed `multiply`,
orthonormal canonical basis, convergent `trace`) **even when the corresponding
"theory" need not exist** — the machinery works whether or not a 4d N=2 theory
sits behind the flow.

The only requirement on `L_a` is that it be a **monomial cone ray**: `L_a^n` a
single auxiliary label (so `E_𝖖(L_a) = Σ_n c_n L_a^n` has a clean single-term
graded tower, `c_n = E_𝖖`-coeff via `sunf_dilog.eq_coeff`).  Any cone generator
of a `ConeKAlgebra` qualifies.

`WildMonopoleRGKAlgebra` — a 't Hooft monopole in pure SU(2)
------------------------------------------------------------
`aux` = the pure-SU(2) cone `PureSU2KAlg`; `S_RG = E_𝖖(L_{1,0})`, the **'t Hooft
monopole** `L_{1,0} = H_0` (cone label `((0,1),)`, a monomial ray: `L_{1,0}^n =
H_0^n = ((0,n),)`).  Graded by the **magnetic charge** (`deg(label) = Σ exps over
H-generators`).  Integrating out a *monopole* hyper is not a standard RG flow —
this is a deliberately wild flow — yet `Tr(1) = 1 + 3q² + 9q⁴ + …` is a clean,
truncation-stable formal series and the basis is orthonormal.

`WildA1D3SquaredRGKAlgebra` — two A1D3's coupled by `E_𝖖(μ L L')`
----------------------------------------------------------------
`aux` = `A1DoddConeKAlg(0) ⊗ A1DoddConeKAlg(0) ⊗ QT_μ([[0]])` (two copies of
`[A₁,D₃]`, each `SU(2)`-flavoured, plus a `U(1)_μ` torus).  `S_RG = E_𝖖(μ·L·L')`
with `L, L'` the **doublet chords** `(((1,1,0),1),0)` of the two factors (each a
monomial ray) — a fictional `μ`-dressed coupling of two AD theories.  Graded by
the μ-charge.  `Tr(1) = 1 + (1 + χ_L² + χ_R²)q² − χ_L χ_R(μ+μ⁻¹)q³ + …` over
`R(SU(2)_L) ⊗ R(SU(2)_R) ⊗ R(U(1)_μ)` — truncation-stable, orthonormal.

Both are **pure** `RGKAlgebra`s (no override), spine-free.  Validation:
the machinery is well-formed — `_fs_exact`
fires, `Tr(1)` is truncation-stable, `I_{1,1}[q⁰] = 1`, `multiply` closes.
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
from a1dodd_kalg import A1DoddConeKAlg
from sunf_dilog import eq_coeff

__all__ = ["WildMonopoleRGKAlgebra", "WildA1D3SquaredRGKAlgebra"]


# ---------------------------------------------------------------------------
# Wild 1 — a 't Hooft monopole in pure SU(2)
# ---------------------------------------------------------------------------
def _magnetic_charge(label) -> int:
    """The magnetic charge of a pure-SU(2) H-tower cone label: `Σ exp` over the
    `H_n` generators (`int` gen = `H_n`, magnetic 1; `('W',n)` = Wilson,
    magnetic 0)."""
    return sum(exp for gen, exp in label if isinstance(gen, int))


class WildMonopoleRGKAlgebra(RGKAlgebra):
    """WILD: pure SU(2) with a (fictional) 't Hooft-monopole hyper integrated out.
    `aux = PureSU2KAlg`, `S_RG = E_𝖖(L_{1,0})` on the magnetic monopole ray,
    graded by magnetic charge.  See the module docstring."""

    def __init__(self) -> None:
        self._aux = PureSU2KAlg()

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        return Grading(rank=1, deg=lambda lbl: (_magnetic_charge(lbl),),
                       height=(1,), cone_gens=((1,),))

    def _s_rg_component(self, p) -> dict:
        """`[S_RG]_{(n,)} = {((0,n),): E_𝖖-coeff(n)}` — the monopole tower
        `L_{1,0}^n = H_0^n` (a single cone monomial)."""
        (n,) = p
        if n < 0:
            return {}
        if n == 0:
            return {self._aux.identity(): eq_coeff(0)}
        return {((0, n),): eq_coeff(n)}

    def rg_generator(self, cutoff: int) -> dict:
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        out: dict = {}
        for n in range(cutoff + 1):
            out.update(self._s_rg_component((n,)))
        return out

    def _section_split(self, label):
        return tuple(label), None

    def __repr__(self) -> str:
        return "WildMonopoleRGKAlgebra(pure SU(2), S_RG=E_𝖖(L_{1,0}) 't Hooft monopole)"


# ---------------------------------------------------------------------------
# Wild 2 — two [A1,D3] coupled by E_𝖖(μ L L')
# ---------------------------------------------------------------------------
def _chord_pow(n: int):
    """The A1D3 doublet chord `L = (((1,1,0),1),0)` to power `n` (a monomial
    ray); the identity `((),0)` at `n = 0`."""
    if n == 0:
        return ((), 0)
    return ((((1, 1, 0), n),), 0)


class WildA1D3SquaredRGKAlgebra(RGKAlgebra):
    """WILD: `[A₁,D₃] ⊗ [A₁,D₃]` coupled by `S_RG = E_𝖖(μ·L·L')` (the doublet
    chords of the two factors, `μ` a `U(1)` fugacity).  `aux = A1DoddConeKAlg(0) ⊗
    A1DoddConeKAlg(0) ⊗ QT_μ([[0]])`, graded by the μ-charge.  See the module
    docstring."""

    def __init__(self) -> None:
        self._a = A1DoddConeKAlg(0)
        self._b = A1DoddConeKAlg(0)
        self._mu = QuantumTorusKAlg([[0]])
        self._aux = TensorKAlgebra(TensorKAlgebra(self._a, self._b), self._mu)

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # μ-charge = the outer QT_μ leg = lbl[1][0].
        return Grading(rank=1, deg=lambda lbl: (lbl[1][0],),
                       height=(1,), cone_gens=((1,),))

    def _s_rg_component(self, p) -> dict:
        """`[S_RG]_{(n,)} = {((L^n, L'^n), (n,)): E_𝖖-coeff(n)}` — the `μ`-dressed
        chord-product tower `(μ L L')^n` (a single label, the chords being
        monomial rays)."""
        (n,) = p
        if n < 0:
            return {}
        if n == 0:
            return {self._aux.identity(): eq_coeff(0)}
        return {((_chord_pow(n), _chord_pow(n)), (n,)): eq_coeff(n)}

    def rg_generator(self, cutoff: int) -> dict:
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        out: dict = {}
        for n in range(cutoff + 1):
            out.update(self._s_rg_component((n,)))
        return out

    def _section_split(self, label):
        return tuple(label), None

    def __repr__(self) -> str:
        return "WildA1D3SquaredRGKAlgebra([A1,D3]⊗[A1,D3], S_RG=E_𝖖(μ L L'))"


if __name__ == "__main__":
    import warnings
    for cls in (WildMonopoleRGKAlgebra, WildA1D3SquaredRGKAlgebra):
        T = cls()
        print(repr(T))
        print("  coeff =", T.coefficient_ring(), " fs_exact =", T._fs_exact_available())
        print("  S_RG levels 0..2:")
        for n in range(3):
            print("    n=%d:" % n, {l: str(c) for l, c in T._s_rg_component((n,)).items()})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tr = T.trace(T.identity(), 4)
            print("  Tr(1):", {e: str(r) for e, r in sorted(tr.coeffs.items())}, " warns =", len(w))
        print()
