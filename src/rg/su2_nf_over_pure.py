"""su2_nf_over_pure — SU(2)+N_f as a **pure exact-FS** `RGKAlgebra` over pure SU(2),
fully BPS-free.  The single-gauge-factor over-pure flow (the bifundamental's
`su2su2_bifund_over_pure` is the two-factor analogue).

    SU(2) + N_f fundamentals   →   pure SU(2)   (the flavours integrated out)

The matter is the spectrum generator, integrated out:

    S_RG = Ψ = ∏_{i=1}^{N_f} E_𝔮(μ_i v) E_𝔮(μ_i / v),

each fundamental's two SU(2) weights `v^{±1}` dressed by the flavour fugacity
`μ_i`, expanded over weights and peeled to pure-SU(2) Wilson characters
`χ_w(v) → F_{0,w}` (the `(0,w)` Wilson sector).  The N_f flavour fugacities are
carried as an `add_flavour(U(1)^{N_f})` coefficient, so the bilinear exact-FS
trace is the **μ-refined** Schur index in `R(U(1)^{N_f})` (the Cartan of the
SU(2)+N_f flavour symmetry SO(2N_f); e.g. N_f=2 gives the SO(4) adjoint
`2 + Σ μ_1^{±1}μ_2^{±1}` = 6 currents at q²).

Auxiliary = `pure SU(2).add_flavour(AbelianZPlusRing(N_f))` — the self-contained
BPS-free pure-SU(2) cone K-algebra (Step 2: `pure_su2_h_cone_data`, trace via
`pure_su2_h_trace_analytic`) with the N_f flavour U(1)s.  Pure exact-FS (RG
solved, no trace override), spine-free.
"""
from __future__ import annotations

import os
import sys
from itertools import product

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import KAlgebra, Element
from rgkalgebra import RGKAlgebra
from grading import Grading
from habiro import HabiroElement
from zplus_ring import AbelianZPlusRing, RPowerSeries
from pure_su2_h_cone_data import PureSU2KAlg
from pure_su2_h_trace_analytic import trace_series
from sunf_dilog import eq_coeff as _a          # a_n = (-1)^n q^n/(q^2;q^2)_n (= E_𝔮 coeff)


def _multi_levels(Nf: int, cutoff: int):
    for k in product(range(cutoff + 1), repeat=Nf):
        if sum(k) <= cutoff:
            yield k


def su2_nf_matter_spectrum(Nf: int, cutoff: int) -> dict[tuple, HabiroElement]:
    """`S_RG = Ψ` truncated to total flavour number `Σ_i k_i ≤ cutoff`.

    Returns `{(0, w, k_1,…,k_{N_f}): c_{k,w}}` — the exact Habiro coefficient of
    the pure-SU(2) Wilson line `χ_w = (0, w)` at flavour multilevel `k`.
    `c_{k,w} = D_k(w) − D_k(w+2)`, the Weyl peel of the scalar v-weight
    polynomial `D_k(W) = Σ ∏_i a_{m_i} a_{n_i}` (`m_i+n_i=k_i`,
    `Σ_i(m_i−n_i)=W`) of `∏_i E_𝔮(μ_i v)E_𝔮(μ_i/v)`."""
    if Nf < 1:
        raise ValueError("Nf must be >= 1")
    out: dict[tuple, HabiroElement] = {}
    for k in _multi_levels(Nf, cutoff):
        Dk = {0: HabiroElement.one()}
        for ki in k:
            terms = {}
            for m in range(ki + 1):
                terms[m - (ki - m)] = _a(m) * _a(ki - m)
            nxt: dict[int, HabiroElement] = {}
            for W1, c1 in Dk.items():
                for W2, c2 in terms.items():
                    c = c1 * c2
                    if c.is_zero():
                        continue
                    nxt[W1 + W2] = nxt.get(W1 + W2, HabiroElement.zero()) + c
            Dk = {W: c for W, c in nxt.items() if not c.is_zero()}
        top = max(Dk.keys(), default=0)
        for w in range(top, -1, -2):
            ckw = Dk.get(w, HabiroElement.zero()) - Dk.get(w + 2, HabiroElement.zero())
            if not ckw.is_zero():
                out[(0, w) + tuple(k)] = ckw
    return out


class _PureSU2KAlgT(KAlgebra):
    """Pure SU(2) cone K-algebra, labels `(m,e)`; multiply via the BPS-free
    `PureSU2KAlg`, trace via the analytic Schur trace `trace_series` (so the
    Wilson `(m,e)` trace works directly — `PureSU2KAlg.trace` operates on native
    H-tower labels).  The N_f flavour is adjoined by `.add_flavour(...)`."""

    def __init__(self) -> None:
        self._c = PureSU2KAlg()

    def coefficient_ring(self):
        return self._c.coefficient_ring()

    def identity(self):
        return (0, 0)

    def multiply(self, a, b):
        return self._c.multiply(a, b)

    def rho(self, a):
        return self._c.rho(a)

    def rho_inverse(self, a):
        return self._c.rho_inverse(a)

    def trace(self, a, K: int = 20):
        lp = trace_series(a[0], a[1], K)
        return RPowerSeries(self.coefficient_ring(), dict(lp._coeffs), K)


class SU2NfOverPure(RGKAlgebra):
    """SU(2)+N_f over `pure SU(2).add_flavour(U(1)^{N_f})`, pure exact-FS; the
    matter `S_RG = ∏_i E_𝔮(μ_i v)E_𝔮(μ_i/v)` peeled to Wilson χ_w, the flavour
    fugacities `μ_i` an `add_flavour` U(1)^{N_f} so the trace is the μ-refined
    (SO(2N_f)-Cartan) Schur index."""

    def __init__(self, Nf: int) -> None:
        self._Nf = int(Nf)
        self._aux = _PureSU2KAlgT().add_flavour(AbelianZPlusRing(self._Nf))

    @property
    def Nf(self) -> int:
        return self._Nf

    def auxiliary(self):
        return self._aux

    def grading(self):
        """`Γ_RG = Z^{N_f}` flavour grading; positive cone the N_f unit axes,
        height `(1,…,1)` (total flavour number)."""
        Nf = self._Nf
        gens = tuple(tuple(1 if i == j else 0 for i in range(Nf)) for j in range(Nf))
        return Grading(rank=Nf, deg=lambda lab: tuple(lab[1]),
                       height=(1,) * Nf, cone_gens=gens)

    def _s_rg_component(self, p):
        """`[Ψ]_p` — exact graded component at flavour multilevel `p`, relabelled
        onto the `add_flavour` auxiliary `((0,w), (k_1,…,k_{N_f}))`; `{}` off the
        positive cone."""
        p = tuple(int(x) for x in p)
        if any(x < 0 for x in p):
            return {}
        K = sum(p)
        return {((0, w), tuple(k)): c
                for (z, w, *k), c in su2_nf_matter_spectrum(self._Nf, K).items()
                if tuple(k) == p}

    def rg_generator(self, cutoff: int) -> dict[tuple, HabiroElement]:
        """`Ψ` windowed to total flavour number `Σ_i k_i ≤ cutoff`."""
        return {((0, w), tuple(k)): c
                for (z, w, *k), c in su2_nf_matter_spectrum(self._Nf, cutoff).items()}

    def _section_split(self, label):
        return tuple(label), None

    def __repr__(self) -> str:
        return f"SU2NfOverPure(N_f={self._Nf})"


if __name__ == "__main__":
    import warnings
    for Nf in (1, 2):
        A = SU2NfOverPure(Nf)
        print(f"SU(2) + N_f={Nf} over pure SU(2):  exact-FS =", A._fs_exact_available())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v = A.trace(A.identity(), 4)
            print("  μ-refined vacuum:",
                  {e: str(r) for e, r in sorted(v.coeffs.items()) if str(r) not in ("0", "")},
                  " warns =", len(w))
