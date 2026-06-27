"""su2_linear_quiver_over_pure — the SU(2)ⁿ linear quiver (a bifundamental on each
link, optional fundamentals at the ends) as a **pure exact-FS** `RGKAlgebra` over
pure SU(2)^⊗ⁿ, fully BPS-free.  The chain generalisation of the over-pure
SU(2)+N_f (`su2_nf_over_pure`) and SU(2)×SU(2) bifundamental
(`su2su2_bifund_over_pure`, the n=2 no-flavour case):

      [N_f⁽¹⁾]──SU(2)₁ ── SU(2)₂ ── ⋯ ── SU(2)_n──[N_f⁽ⁿ⁾]   →   SU(2)ⁿ

(the n−1 link bifundamentals + end fundamentals integrated out).

  * S_RG = Ψ = ∏ link bifundamentals · ∏ end fundamentals, each factor
    `∏ E_𝔮(μ · …)` peeled to SU(2)ⁿ Wilson characters `χ_w → F_{0,w}`; factors
    sharing a node combine by SU(2) Clebsch–Gordan, each carrying its own
    μ-grading slot.  The matter content reuses the bifundamental
    (`su2su2_bifund_matter_spectrum`) and end-fundamental (`su2_nf_matter_spectrum`)
    spectra.
  * aux = `(pure SU(2)^⊗ⁿ).add_flavour(U(1)^L)`, `L = (n−1)+Nf₁+Nf_n` — `n`
    decoupled pure-SU(2) cone factors (trace = product of the BPS-free analytic
    Schur traces) with the `L` matter fugacities as ring coefficients.  So the
    bilinear exact-FS trace is the **μ-refined** Schur index in `R(U(1)^L)`.
  * `Γ_RG` = `Z^L` (link levels then end-fundamental levels); positive cone the
    `L` axes; height = total μ-number.

End-flavour bound: each SU(2) node admits `N_f ≤ 4`; an end node already sees one
bifundamental (= 2), so `Nf₁, Nf_n ≤ 2`.  Pure exact-FS, spine-free.
"""
from __future__ import annotations

import os
import sys

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
from su2su2_bifund_over_pure import su2su2_bifund_matter_spectrum
from su2_nf_over_pure import su2_nf_matter_spectrum

_MAX_END_FLAVOURS = 2          # SU(2) flavour bound: end node = bifund(2) + N_f ≤ 4.


# ---- matter spectrum: character-ring convolution with Clebsch–Gordan ----

def _cg(a: int, b: int) -> list[int]:
    """SU(2) Clebsch–Gordan `χ_a ⊗ χ_b = ⊕_c χ_c` (`c = |a−b|,…,a+b`)."""
    return list(range(abs(a - b), a + b + 1, 2))


def _prune(table: dict) -> dict:
    out = {}
    for w, levmap in table.items():
        lm = {lev: c for lev, c in levmap.items() if not c.is_zero()}
        if lm:
            out[w] = lm
    return out


def _fold_node(running: dict, node: int, spec1d: dict, off: int, cutoff: int) -> dict:
    nxt: dict = {}
    for weights, levmap in running.items():
        a = weights[node]
        for (w, sub), c in spec1d.items():
            for c2 in _cg(a, w):
                nw = list(weights)
                nw[node] = c2
                nw = tuple(nw)
                slot = nxt.setdefault(nw, {})
                for lev, coeff in levmap.items():
                    nl = list(lev)
                    for j, x in enumerate(sub):
                        nl[off + j] += x
                    if sum(nl) > cutoff:
                        continue
                    nl = tuple(nl)
                    v = coeff * c
                    slot[nl] = slot.get(nl, HabiroElement.zero()) + v
    return _prune(nxt)


def _fold_link(running: dict, i: int, link: dict, off: int, cutoff: int) -> dict:
    nxt: dict = {}
    for weights, levmap in running.items():
        a, b = weights[i], weights[i + 1]
        for (wL, wR, k), c in link.items():
            for cL in _cg(a, wL):
                for cR in _cg(b, wR):
                    nw = list(weights)
                    nw[i] = cL
                    nw[i + 1] = cR
                    nw = tuple(nw)
                    slot = nxt.setdefault(nw, {})
                    for lev, coeff in levmap.items():
                        nl = list(lev)
                        nl[off] += k
                        if sum(nl) > cutoff:
                            continue
                        nl = tuple(nl)
                        v = coeff * c
                        slot[nl] = slot.get(nl, HabiroElement.zero()) + v
    return _prune(nxt)


def _end_spectrum(Nf: int, cutoff: int) -> dict:
    spec = {}
    for lab, c in su2_nf_matter_spectrum(Nf, cutoff).items():
        spec[(lab[1], tuple(lab[2:]))] = c
    return spec


def su2_linear_quiver_matter_spectrum(n: int, cutoff: int,
                                      Nf1: int = 0, Nfn: int = 0) -> dict[tuple, HabiroElement]:
    """`S_RG = Ψ` for the SU(2)ⁿ linear quiver (`Nf1`/`Nfn` end fundamentals),
    truncated to total μ-number ≤ `cutoff`.  Labels
    `(0,w₁,…,0,w_n, f₁,…,f_L)`, `L=(n−1)+Nf1+Nfn`, slots ordered
    `[link₁,…,link_{n−1}, end1…, endn…]`."""
    if n < 2:
        raise ValueError("linear quiver needs n >= 2 nodes")
    if not (0 <= Nf1 <= _MAX_END_FLAVOURS and 0 <= Nfn <= _MAX_END_FLAVOURS):
        raise ValueError(f"end fundamentals capped at {_MAX_END_FLAVOURS}; got {Nf1},{Nfn}")
    L = (n - 1) + Nf1 + Nfn
    link = {(wL, wR, k): c
            for (_z1, wL, _z2, wR, k), c in su2su2_bifund_matter_spectrum(cutoff).items()}
    running: dict = {(0,) * n: {(0,) * L: HabiroElement.one()}}
    if Nf1:
        running = _fold_node(running, 0, _end_spectrum(Nf1, cutoff), off=n - 1, cutoff=cutoff)
    for i in range(n - 1):
        running = _fold_link(running, i, link, off=i, cutoff=cutoff)
    if Nfn:
        running = _fold_node(running, n - 1, _end_spectrum(Nfn, cutoff),
                             off=n - 1 + Nf1, cutoff=cutoff)
    out: dict[tuple, HabiroElement] = {}
    for weights, levmap in running.items():
        lab_w = tuple(x for w in weights for x in (0, w))
        for lev, c in levmap.items():
            if not c.is_zero():
                out[lab_w + lev] = c
    return out


# ---- auxiliary: pure SU(2)^⊗n (the U(1)^L flavour is via add_flavour) ----

class _PureSU2nKAlgT(KAlgebra):
    """`n` decoupled pure-SU(2) cone K-algebras tensored.  Labels
    `(m₁,e₁,…,m_n,e_n)`; multiply tensors the `n` BPS-free cone multiplies, trace
    is the product of the per-factor analytic Schur traces `trace_series`."""

    def __init__(self, n: int) -> None:
        self._n = int(n)
        self._cones = [PureSU2KAlg() for _ in range(self._n)]

    def _gauge(self, label, j):
        return (label[2 * j], label[2 * j + 1])

    def coefficient_ring(self):
        return self._cones[0].coefficient_ring()

    def identity(self):
        return (0,) * (2 * self._n)

    def multiply(self, a, b):
        # tensor (cartesian product) of the n per-factor cone multiplies
        terms = [list(self._cones[j].multiply(self._gauge(a, j),
                                              self._gauge(b, j)).terms.items())
                 for j in range(self._n)]
        out: dict = {}

        def rec(j, lbl, coeff):
            if j == self._n:
                out[tuple(lbl)] = coeff
                return
            for (M, E), c in terms[j]:
                rec(j + 1, lbl + [M, E], c if coeff is None else coeff * c)

        rec(0, [], None)
        return Element({k: v for k, v in out.items() if v is not None and not v.is_zero()})

    def rho(self, a):
        out = []
        for j in range(self._n):
            r = self._cones[j].rho(self._gauge(a, j))
            out += [r[0], r[1]]
        return tuple(out)

    def rho_inverse(self, a):
        out = []
        for j in range(self._n):
            r = self._cones[j].rho_inverse(self._gauge(a, j))
            out += [r[0], r[1]]
        return tuple(out)

    def trace(self, a, K: int = 20):
        prod = trace_series(a[0], a[1], K)
        for j in range(1, self._n):
            prod = prod * trace_series(a[2 * j], a[2 * j + 1], K)
        return RPowerSeries(self.coefficient_ring(), dict(prod._coeffs), K)


class SU2LinearQuiverOverPure(RGKAlgebra):
    """SU(2)ⁿ linear quiver over `(pure SU(2)^⊗ⁿ).add_flavour(U(1)^L)`, pure
    exact-FS; the chain generalisation of `SU2NfOverPure` / `SU2xSU2BifundOverPure`."""

    def __init__(self, n: int, Nf1: int = 0, Nfn: int = 0) -> None:
        if n < 2:
            raise ValueError("linear quiver needs n >= 2 nodes")
        self._n = int(n)
        self._Nf1 = int(Nf1)
        self._Nfn = int(Nfn)
        self._L = (self._n - 1) + self._Nf1 + self._Nfn
        self._aux = _PureSU2nKAlgT(self._n).add_flavour(AbelianZPlusRing(self._L))

    @property
    def n(self) -> int:
        return self._n

    def auxiliary(self):
        return self._aux

    def grading(self):
        """`Γ_RG = Z^L` matter grading; positive cone the `L` axes, height
        `(1,…,1)`."""
        L = self._L
        gens = tuple(tuple(1 if i == j else 0 for i in range(L)) for j in range(L))
        return Grading(rank=L, deg=lambda lab: tuple(lab[1]),
                       height=(1,) * L, cone_gens=gens)

    def _s_rg_component(self, p):
        """`[Ψ]_p` — exact graded component at matter multilevel `p`, relabelled
        onto the `add_flavour` auxiliary `((gauge…), (f₁,…,f_L))`; `{}` off-cone."""
        p = tuple(int(x) for x in p)
        if any(x < 0 for x in p):
            return {}
        K = sum(p)
        nn = 2 * self._n
        return {(tuple(lab[:nn]), tuple(lab[nn:])): c
                for lab, c in su2_linear_quiver_matter_spectrum(
                    self._n, K, self._Nf1, self._Nfn).items()
                if tuple(lab[nn:]) == p}

    def rg_generator(self, cutoff: int) -> dict[tuple, HabiroElement]:
        """`Ψ` windowed to total μ-number ≤ `cutoff`."""
        nn = 2 * self._n
        return {(tuple(lab[:nn]), tuple(lab[nn:])): c
                for lab, c in su2_linear_quiver_matter_spectrum(
                    self._n, cutoff, self._Nf1, self._Nfn).items()}

    def _section_split(self, label):
        return tuple(label), None

    def __repr__(self) -> str:
        return f"SU2LinearQuiverOverPure(n={self._n}, Nf1={self._Nf1}, Nfn={self._Nfn})"


if __name__ == "__main__":
    import warnings
    for (n, Nf1, Nfn) in [(2, 0, 0), (3, 0, 0), (2, 1, 0)]:
        A = SU2LinearQuiverOverPure(n, Nf1, Nfn)
        print(repr(A), " exact-FS =", A._fs_exact_available())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v = A.trace(A.identity(), 3)
            print("  μ-refined vacuum (K3):",
                  {e: str(r) for e, r in sorted(v.coeffs.items()) if str(r) not in ("0", "")},
                  " warns =", len(w))
