"""su2_linear_quiver_over_pure — the SU(2)ⁿ linear quiver (bifundamental on each
link, optional fundamentals at the ends) as an `RGKAlgebra` wrapping pure
SU(2)^⊗ⁿ, fully BPS-free.

The chain generalisation of `su2su2_bifund_over_pure.SU2xSU2BifundOverPure`
(which is the `n = 2`, no-flavour case): build the linear quiver

      [N_f⁽¹⁾]──SU(2)₁ ── SU(2)₂ ── ⋯ ── SU(2)_n──[N_f⁽ⁿ⁾]   (n−1 bifund links)

on top of the (decoupled) product of `n` self-contained pure-SU(2) cone
K-algebras, by supplying the matter spectrum in closed form.

  * IR auxiliary = `(pure SU(2)₁ ⊗ ⋯ ⊗ pure SU(2)_n).add_flavour(U(1)^L)`
    (`_PureSU2nKAlg` + `add_flavour`), `L = (n−1) + N_f⁽¹⁾ + N_f⁽ⁿ⁾`: gauge labels
    `(m₁,e₁,…,m_n,e_n)`; the gauge factors are decoupled, so multiply is the tensor
    of the `n` cone multiplies (BPS-free, `PureSU2KAlg`).  The `L` matter μ-levels
    are adjoined as a genuine **coefficient-ring flavour** `R(U(1)^L)` (not baked
    into the labels), so the trace keeps the μ-character — the μ-refined Schur
    index — exactly as `SU2xSU2BifundOverPure` does (a central-level encoding
    cannot, and its trace raises on flavour-charged states rather than
    silently degrading).
  * S_RG = the product of all matter factors

        Ψ = ∏_{a=1}^{N_f⁽¹⁾} F⁽¹⁾_a(v₁) · ∏_{i=1}^{n−1} Ψ_i · ∏_{b=1}^{N_f⁽ⁿ⁾} F⁽ⁿ⁾_b(v_n),

    each link bifundamental `Ψ_i = ∏_{ε,ε'} E_𝔮(μ_i v_i^{ε} v_{i+1}^{ε'})` and
    each end fundamental `F_a(v) = E_𝔮(μ_a v) E_𝔮(μ_a / v)`, re-expressed in
    SU(2)ⁿ characters with `χ_w → F_{0,w}`.  Factors sharing a node combine by
    SU(2) Clebsch–Gordan (`χ_a · χ_w = ⊕_c χ_c`); each factor carries its own
    μ-grading.
  * `Γ_RG` grading = `Z^L` (`n−1` link levels, then `N_f⁽¹⁾` left- and `N_f⁽ⁿ⁾`
    right-end fundamental levels); height = total μ-number.  Tame (abelian,
    central) — the flow integrates out every hyper, leaving pure SU(2)ⁿ.

End-flavour bound: each SU(2) node admits `N_f ≤ 4`.  An end node already sees
one bifundamental (= 2 flavours), so it takes **up to 2** fundamentals; internal
nodes see two bifundamentals (= 4) and take none.  Hence `N_f⁽¹⁾, N_f⁽ⁿ⁾ ≤ 2`.

Fully BPS-free: link content reuses `su2su2_bifund_over_pure`, end content reuses
`su2_nf_over_pure` (both `HabiroElement.nahm_term` coefficients), and the
auxiliary multiply/trace route through `PureSU2KAlg` / the analytic
Schur trace — no BPS peel.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import KAlgebra, Element
from rgkalgebra import RGKAlgebra
from grading import Grading
from habiro import HabiroElement
from zplus_ring import AbelianZPlusRing, RPowerSeries
from pure_su2_h_cone_data import PureSU2KAlg
from pure_su2_h_trace_analytic import trace_series       # BPS-free Wilson/Schur trace
from su2su2_bifund_over_pure import su2su2_bifund_matter_spectrum
from su2_nf_over_pure import su2_nf_matter_spectrum

_MAX_END_FLAVOURS = 2          # SU(2) flavour bound: end node = bifund(2) + N_f ≤ 4.


# ---------------------------------------------------------------------------
# Matter spectrum  Ψ = (end₁ fundamentals) · (chain bifundamentals) · (end_n
# fundamentals)  — a character-ring convolution with Clebsch–Gordan on every
# shared node, each factor carrying its own μ-grading in a fixed slot.
# ---------------------------------------------------------------------------


def _cg(a: int, b: int) -> list[int]:
    """SU(2) Clebsch–Gordan tensor decomposition `χ_a ⊗ χ_b = ⊕_c χ_c`
    (`c = |a−b|, |a−b|+2, …, a+b`)."""
    return list(range(abs(a - b), a + b + 1, 2))


def _prune(table: dict) -> dict:
    """Drop exact-zero coefficients and emptied node-weight rows."""
    out = {}
    for w, levmap in table.items():
        lm = {lev: c for lev, c in levmap.items() if not c.is_zero()}
        if lm:
            out[w] = lm
    return out


def _fold_node(running: dict, node: int, spec1d: dict, off: int,
               cutoff: int) -> dict:
    """Multiply the running character product by a single-node factor `spec1d =
    {(w, sub_levels): coeff}` living on `node` (its μ-levels go into the
    full-length multilevel starting at offset `off`), combining the node content
    by SU(2) Clebsch–Gordan."""
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
    """Multiply the running product by the bifundamental on link `i` (nodes
    `i, i+1`), `link = {(wL, wR, k): coeff}`, combining *both* shared nodes by
    Clebsch–Gordan (robust to either node already carrying content); the link
    level `k` goes into the multilevel slot `off`."""
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
    """Single-node `N_f`-fundamental content `{(w, (k₁,…,k_{N_f})): coeff}`
    (the `SU2NfOverPure` matter `∏ E_𝔮(μ_a v)E_𝔮(μ_a/v)`)."""
    spec = {}
    for lab, c in su2_nf_matter_spectrum(Nf, cutoff).items():
        spec[(lab[1], tuple(lab[2:]))] = c
    return spec


def su2_linear_quiver_matter_spectrum(n: int, cutoff: int,
                                      Nf1: int = 0, Nfn: int = 0) -> dict[tuple, HabiroElement]:
    """`S_RG = Ψ` for the SU(2)ⁿ linear quiver with `Nf1`/`Nfn` fundamentals on
    the left/right end nodes, truncated to total μ-number `Σ levels ≤ cutoff`.

    Returns `{(0,w₁,0,w₂,…,0,w_n, f₁,…,f_L): c}`, `L = (n−1)+Nf1+Nfn`, the exact
    `HabiroElement` coefficient of `∏_j χ_{w_j}(v_j)` at multilevel `(f₁,…,f_L)`
    ordered `[link₁,…,link_{n−1}, end1₁,…,end1_{Nf1}, endn₁,…,endn_{Nfn}]`.
    """
    if n < 2:
        raise ValueError("linear quiver needs n >= 2 nodes (n-1 >= 1 links)")
    if not (0 <= Nf1 <= _MAX_END_FLAVOURS and 0 <= Nfn <= _MAX_END_FLAVOURS):
        raise ValueError(
            f"end fundamentals capped at {_MAX_END_FLAVOURS} (SU(2) N_f<=4, "
            f"bifund = 2); got Nf1={Nf1}, Nfn={Nfn}")
    L = (n - 1) + Nf1 + Nfn

    link = {(wL, wR, k): c
            for (_z1, wL, _z2, wR, k), c in su2su2_bifund_matter_spectrum(cutoff).items()}

    running: dict = {(0,) * n: {(0,) * L: HabiroElement.one()}}
    # left-end fundamentals (node 0), slots [n-1, n-1+Nf1).
    if Nf1:
        running = _fold_node(running, 0, _end_spectrum(Nf1, cutoff),
                             off=n - 1, cutoff=cutoff)
    # chain bifundamentals, link i -> slot i.
    for i in range(n - 1):
        running = _fold_link(running, i, link, off=i, cutoff=cutoff)
    # right-end fundamentals (node n-1), slots [n-1+Nf1, L).
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


# ---------------------------------------------------------------------------
# Auxiliary: pure SU(2)^⊗n ⊗ U(1)^L flavour.
# ---------------------------------------------------------------------------


class _PureSU2nKAlg(KAlgebra):
    """`n` decoupled pure-SU(2) cone K-algebras tensored.  Labels
    `(m₁,e₁,…,m_n,e_n)`; multiply tensors the `n` BPS-free `PureSU2KAlg` cone
    multiplies (the gauge factors q-commute trivially), trace is the product of
    the per-factor analytic Schur traces `trace_series` (the `(m,e)` Wilson/'t
    Hooft trace, BPS-free).

    The `L` matter μ-levels are **not** baked in here — they are adjoined by
    `.add_flavour(AbelianZPlusRing(L))`, so the coefficient ring carries the
    μ-fugacities and the (generic) trace keeps the μ-refined index (a
    central-"level" encoding cannot, and its trace has to raise on
    flavour-charged states rather than silently degrade — the same design
    choice as `SU2xSU2BifundOverPure`)."""

    def __init__(self, n: int) -> None:
        if n < 2:
            raise ValueError("need n >= 2 nodes")
        self._n = int(n)
        self._cones = [PureSU2KAlg() for _ in range(self._n)]

    def _gauge(self, label, j):
        return (label[2 * j], label[2 * j + 1])

    def coefficient_ring(self):
        return self._cones[0].coefficient_ring()

    def identity(self):
        return (0,) * (2 * self._n)

    def multiply(self, a, b):
        n = self._n
        node_terms = [list(self._cones[j].multiply(self._gauge(a, j),
                                                   self._gauge(b, j)).terms.items())
                      for j in range(n)]
        out: dict = {}

        def rec(j, lbl, coeff):
            if j == n:
                out[tuple(lbl)] = coeff
                return
            for (M, E), c in node_terms[j]:
                rec(j + 1, lbl + [M, E], c if coeff is None else coeff * c)

        rec(0, [], None)
        return Element({k: v for k, v in out.items()
                        if v is not None and not v.is_zero()})

    def rho(self, a):
        gs = [self._cones[j].rho(self._gauge(a, j)) for j in range(self._n)]
        return tuple(x for g in gs for x in g)

    def rho_inverse(self, a):
        gs = [self._cones[j].rho_inverse(self._gauge(a, j)) for j in range(self._n)]
        return tuple(x for g in gs for x in g)

    def trace(self, a, K=20):
        # Decoupled tensor product: Tr(⊗_j x_j) = ∏_j Tr(x_j); per-factor trace is
        # the BPS-free analytic Schur trace `trace_series` on the (m,e) label.
        prod = None
        for j in range(self._n):
            m, e = self._gauge(a, j)
            lp = trace_series(m, e, K)
            prod = lp if prod is None else prod * lp
        return RPowerSeries(self.coefficient_ring(), dict(prod._coeffs), K)


# ---------------------------------------------------------------------------
# The RGKAlgebra.
# ---------------------------------------------------------------------------


class SU2LinearQuiverOverPure(RGKAlgebra):
    """The SU(2)ⁿ linear quiver — bifundamental on each link, optional
    fundamentals (`Nf1`/`Nfn ≤ 2`) on the end nodes — on pure SU(2)^⊗ⁿ, fully
    BPS-free; the chain generalisation of `SU2xSU2BifundOverPure`."""

    def __init__(self, n: int, Nf1: int = 0, Nfn: int = 0) -> None:
        if n < 2:
            raise ValueError("linear quiver needs n >= 2 nodes")
        if not (0 <= Nf1 <= _MAX_END_FLAVOURS and 0 <= Nfn <= _MAX_END_FLAVOURS):
            raise ValueError(
                f"end fundamentals capped at {_MAX_END_FLAVOURS}; "
                f"got Nf1={Nf1}, Nfn={Nfn}")
        self._n = int(n)
        self._Nf1 = int(Nf1)
        self._Nfn = int(Nfn)
        self._L = (self._n - 1) + self._Nf1 + self._Nfn
        self._aux = _PureSU2nKAlg(self._n).add_flavour(AbelianZPlusRing(self._L))

    @property
    def n(self) -> int:
        return self._n

    @property
    def end_flavours(self) -> tuple[int, int]:
        return (self._Nf1, self._Nfn)

    def auxiliary(self):
        return self._aux

    def grading(self):
        """`Γ_RG = Z^L` (`L = (n−1)+Nf1+Nfn`) = the `add_flavour` μ-levels;
        height `(1,…,1)` (total μ-number).

        The positive cone is the **non-negative orthant** — every matter factor
        is an `E_𝔮` whose expansion carries only non-negative powers of its own
        μ-level, so every appearing charge has `p_i ≥ 0` (matching the
        `_s_rg_component` cone, which vanishes when any `p_i < 0`).  Declaring it
        via the `L` unit rays enables the exact per-η FS oracle
        (`_fs_exact_available`), so the μ-refined trace is computed exactly — the
        same path the `n = 2` `SU2xSU2BifundOverPure` already uses (`L = 1`,
        `cone_gens = ((1,),)`)."""
        L = self._L
        cone_gens = tuple(
            tuple(1 if i == j else 0 for j in range(L)) for i in range(L)
        )
        return Grading(rank=L, deg=lambda lab: tuple(lab[1]),
                       height=(1,) * L, cone_gens=cone_gens)

    def _s_rg_component(self, p):
        """`[Ψ]_p` — exact graded component at matter multilevel `p`, relabelled
        onto the `add_flavour` auxiliary `((m₁,e₁,…,m_n,e_n), (f₁,…,f_L))`; `{}`
        off the cone (`p_i < 0`)."""
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
        """`Ψ` windowed to total μ-number `Σ levels ≤ cutoff`, relabelled onto the
        `add_flavour` auxiliary `((m₁,e₁,…,m_n,e_n), (f₁,…,f_L))`."""
        nn = 2 * self._n
        return {(tuple(lab[:nn]), tuple(lab[nn:])): c
                for lab, c in su2_linear_quiver_matter_spectrum(
                    self._n, cutoff, self._Nf1, self._Nfn).items()}

    def _section_split(self, label):
        """Auxiliary labels are `((m₁,e₁,…,m_n,e_n), (f₁,…,f_L))` — the gauge
        tensor is the section, the matter levels `(f…)` the (central, additive)
        flavour; the SU(2) Wilson content fuses by Clebsch–Gordan inside the
        section, so disable the flavour-shift multiply cache (`flav = None`) and
        let the generic `from_ir_image(RG(a)·RG(b))` route through `PureSU2KAlg`
        (mirrors `SU2xSU2BifundOverPure`)."""
        return tuple(label), None

    def __repr__(self) -> str:
        return (f"SU2LinearQuiverOverPure(n={self._n}, "
                f"Nf1={self._Nf1}, Nfn={self._Nfn})")


# ---------------------------------------------------------------------------
# Demonstration.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [(2, 0, 0), (3, 0, 0), (2, 1, 1), (3, 2, 2)]
    for (n, Nf1, Nfn) in cases:
        A = SU2LinearQuiverOverPure(n, Nf1, Nfn)
        tag = f"SU(2)^{n} linear quiver, end flavours ({Nf1},{Nfn})"
        print(f"==============  {tag}  ==============")
        S = A.rg_generator(2)
        g = A.grading()
        print(f"  Γ_RG rank {g.rank} (= {n-1} links + {Nf1} + {Nfn} ends); "
              f"S_RG (Σ level ≤ 2): {len(S)} terms")
        for lab in sorted(S, key=lambda t: (sum(t[1]), t[0][1::2])):
            print(f"    {lab}:  {S[lab]}")
        print()
