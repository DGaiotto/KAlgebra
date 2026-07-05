"""quiver_over_pure — linear U(N₁)×⋯×U(N_n) quiver gauge theories as
`RGKAlgebra`s wrapping `⊗_a` pure U(N_a): the generic `QuiverOverPure`
class (Plan 22 T3; D3 ruling 2026-06-10 — one generic class, nodes =
`PureUNKAlgebra`s, edges = bifundamental `S_RG` factors; the SU(2)ⁿ
prototypes `su2su2_bifund_over_pure` / `su2_linear_quiver_over_pure` stay
as per-theory reference implementations).

The recipe (the standard one — cf. `un_nf_over_pure_rgflow`):

  1. auxiliary = `⊗_a PureUNKAlgebra(N_a)` (decoupled tensor,
     `_TensorPureUN`), promoted by `add_flavour(AbelianZPlusRing(L))` with
     one μ-level slot per link and per fundamental flavour — labels
     `((lab₁,…,lab_n), k_vec)` (the D2 pair-label pattern);
  2. `S_RG = Ψ = ∏_{i<n} Ψ_i · ∏_a ∏_{f≤Nf_a} F_{a,f}` — the link
     bifundamental `(N_i, N̄_{i+1})` spectrum

         Ψ_i = ∏_{j≤N_i} ∏_{l≤N_{i+1}} E_𝔮(μ_i · v⁽ⁱ⁾_j / v⁽ⁱ⁺¹⁾_l)

     and per-node fundamentals `F_{a,f} = ∏_j E_𝔮(μ_{a,f}·v⁽ᵃ⁾_j)`,
     expanded on per-node Wilson characters: the link's level-k component
     is the **double-Schur expansion** `{(λ, ν): c}` (inverse Kostka on
     both sides; `bifund_pair_expansion`), the node-(i+1) content sitting
     in inverse powers — `s_ν(v⁻¹) = det^{−ν₁}·χ_{λ′}`, a det-twisted
     dominant weight.  Factors sharing a node combine by the pure Wilson
     product (exact classical LR at q⁰).
  3. `Γ_RG = Z^L` by μ-levels (height = total hyper number; one cone
     generator per slot).

For U(N) nodes the link U(1) sits inside the two gauge centres, so μ_i is
a *formal* (redundant-but-consistent) refinement: it counts link-hyper
quanta — exactly the tame grading the RG solve needs.  The SU(2)ⁿ
prototypes' honest flavour μ is the same device with the centres
ungauged.  Matter corrections are O(𝔮) term-by-term
(`a_n = (−1)ⁿqⁿ/(q²;q²)_n`), so `RG(a) = a + O(𝔮)` holds in the physical
q-grading independently of the μ bookkeeping.

`trace` / `inner_product` are the chart-evaluated §6b FS pairing of
`un_nf_over_pure_rgflow`, lifted to **sums of pure-URQTorus tensors**
(speed-only override; the generic transport stays reachable via `super()`
and is the cross-validation reference).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fractions import Fraction
from itertools import product

from kalgebra import KAlgebra, Element
from rgkalgebra import RGKAlgebra
from grading import Grading
from habiro import HabiroElement
from zplus_ring import AbelianZPlusRing, RElement, RPowerSeries

from pure_un_kalgebra import PureUNKAlgebra, L_W, _lowest_charge
from un_nf_over_pure_rgflow import (
    _a, _inv_kostka, _pure_un_dressed_rays, single_hyper_wilson_expansion,
)


# ---------------------------------------------------------------------------
# The bifundamental spectrum: ∏_{j,l} E_𝔮(μ x_j w_l) on Schur × Schur.
# ---------------------------------------------------------------------------


def _compositions(total: int, cells: int):
    """All length-`cells` tuples of non-negative ints summing to `total`."""
    if cells == 0:
        if total == 0:
            yield ()
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, cells - 1):
            yield (first,) + rest


def bifund_pair_expansion(Na: int, Nb: int, k: int) -> dict[tuple, HabiroElement]:
    """`[∏_{j≤Na} ∏_{l≤Nb} E_𝔮(μ·x_j·w_l)]_{μ^k}` as `{(λ, ν): c}` over
    `s_λ(x)·s_ν(w)` (exact Habiro; partitions without trailing zeros).

    For the bifundamental `(N_a, N̄_b)` put `w = 1/v⁽ᵇ⁾`: the node-b factor
    is `s_ν(v⁻¹) = det^{−ν₁}·χ_{λ′}` with `λ′ = ν₁·(1^{N_b}) − rev(ν)`.

    The monomial table `P(ρ, γ)` (coefficient of `x^ρ w^γ` at dominant
    `ρ, γ`) is summed over non-negative `Na×Nb` matrices with that exact
    (ordered) row/column-sum pair; both sides then convert monomial → Schur
    by the inverse Kostka matrices (integral)."""
    parts_a, Inv_a = _inv_kostka(k, Na)
    parts_b, Inv_b = _inv_kostka(k, Nb)
    Z = HabiroElement.zero()

    P: dict[tuple, HabiroElement] = {}
    for n in _compositions(k, Na * Nb):
        rows = tuple(sum(n[j * Nb:(j + 1) * Nb]) for j in range(Na))
        cols = tuple(sum(n[j * Nb + l] for j in range(Na)) for l in range(Nb))
        if any(rows[i] < rows[i + 1] for i in range(Na - 1)):
            continue
        if any(cols[i] < cols[i + 1] for i in range(Nb - 1)):
            continue
        c = HabiroElement.one()
        for x in n:
            if x:
                c = c * _a(x)
        key = (tuple(x for x in rows if x), tuple(x for x in cols if x))
        P[key] = P.get(key, Z) + c

    out: dict[tuple, HabiroElement] = {}
    for ia, lam in enumerate(parts_a):
        for ib, nu in enumerate(parts_b):
            acc = Z
            for ja, rho in enumerate(parts_a):
                fa = Inv_a[ja][ia]
                if fa == 0:
                    continue
                for jb, gam in enumerate(parts_b):
                    fb = Inv_b[jb][ib]
                    if fb == 0:
                        continue
                    Pq = P.get((rho, gam))
                    if Pq is None:
                        continue
                    f = fa * fb
                    assert f.denominator == 1, (rho, gam, f)
                    acc = acc + Pq * int(f)
            if not acc.is_zero():
                out[(lam, nu)] = acc
    return out


# ---------------------------------------------------------------------------
# Auxiliary: ⊗_a pure U(N_a)  (decoupled tensor; flavour is added on top by
# the standard `add_flavour`).
# ---------------------------------------------------------------------------


class _TensorPureUN(KAlgebra):
    """Decoupled tensor of pure-U(N_a) K-algebras.  Labels are tuples of
    per-node `(m, e)` engine labels; multiply / ρ / trace are node-wise
    (`⟨·,·⟩ = 0` across nodes, so the gauge factors q-commute trivially)."""

    def __init__(self, pures) -> None:
        self._ps = list(pures)
        if not self._ps:
            raise ValueError("need at least one node")

    def nodes(self):
        return list(self._ps)

    def coefficient_ring(self):
        return self._ps[0].coefficient_ring()

    def identity(self):
        return tuple(P.identity() for P in self._ps)

    def multiply(self, a, b):
        per = [P.multiply(a[j], b[j]) for j, P in enumerate(self._ps)]
        out: dict = {}
        for combo in product(*[list(p.terms.items()) for p in per]):
            lab = tuple(l for (l, _c) in combo)
            c = None
            for (_l, cj) in combo:
                c = cj if c is None else c * cj
            out[lab] = c
        return Element(out)

    def rho(self, a):
        return tuple(P.rho(a[j]) for j, P in enumerate(self._ps))

    def rho_inverse(self, a):
        return tuple(P.rho_inverse(a[j]) for j, P in enumerate(self._ps))

    def trace(self, a, K: int = 20) -> RPowerSeries:
        # Decoupled tensor: Tr(⊗_j x_j) = ∏_j Tr(x_j) (integer q-series).
        acc = {0: 1}
        for j, P in enumerate(self._ps):
            tj = P.trace(a[j], K)
            nxt: dict = {}
            for e1, c1 in acc.items():
                for e2, c2 in tj.coeffs.items():
                    e = e1 + e2
                    if e > K:
                        continue
                    nxt[e] = nxt.get(e, 0) + c1 * c2
            acc = {e: c for e, c in nxt.items() if c}
        return RPowerSeries(self.coefficient_ring(), acc, K)

    def _label_section_decompose(self, label):
        # Trivial section; when the coefficient ring is trivial the lift
        # coordinate is inherited from the KAlgebra universal treatment.
        return (label, self.coefficient_ring().one())


# ---------------------------------------------------------------------------
# The RGKAlgebra.
# ---------------------------------------------------------------------------


class QuiverOverPure(RGKAlgebra):
    """The linear quiver `[Nf₁]—U(N₁)—U(N₂)—⋯—U(N_n)—…—[Nf_n]` (one
    bifundamental per link, `Nf_a` fundamentals at node `a`) over
    `⊗_a PureUNKAlgebra(N_a)`, by the standard recipe: `add_flavour` +
    `S_RG = Ψ`.  The generic `RGKAlgebra` derives RG / multiply / trace /
    inner_product; `trace`/`inner_product` carry the chart-evaluated §6b
    speed override (tensor charts)."""

    def __init__(self, ranks, Nf=None, dress: int = 1, max_len: int = 2,
                 wilson_deg: int = 4, K: int = 8) -> None:
        self._ranks = tuple(int(N) for N in ranks)
        n = len(self._ranks)
        if n < 1 or any(N < 1 for N in self._ranks):
            raise ValueError(f"bad ranks {ranks}")
        self._Nfs = tuple(int(x) for x in (Nf if Nf is not None else (0,) * n))
        if len(self._Nfs) != n or any(x < 0 for x in self._Nfs):
            raise ValueError(f"bad per-node flavour counts {Nf}")
        self._L = (n - 1) + sum(self._Nfs)
        if self._L < 1:
            raise ValueError(
                "no matter (single node, Nf=0) — use PureUNKAlgebra directly")
        self._pures = [
            PureUNKAlgebra(N, _pure_un_dressed_rays(N, dress),
                           max_len=max_len, K=K, wilson_deg=wilson_deg)
            for N in self._ranks
        ]
        self._base = _TensorPureUN(self._pures)
        self._aux = self._base.add_flavour(AbelianZPlusRing(rank=self._L))
        self._wl: dict = {}          # (node, λ) → Wilson gauge label
        self._wlinv: dict = {}       # (node, ν) → label of s_ν(v⁻¹)
        self._edge_cache: dict = {}  # (link, k) → {(g_a, g_b): Habiro}
        self._fuse_cache: dict = {}  # (node, g, h) → ((label, int LR), …)

    # ----- shape ----------------------------------------------------------

    @property
    def ranks(self) -> tuple:
        return self._ranks

    @property
    def n_nodes(self) -> int:
        return len(self._ranks)

    @property
    def flavours(self) -> tuple:
        return self._Nfs

    def pures(self):
        return list(self._pures)

    def _flavour_slot(self, a: int, f: int) -> int:
        """Multilevel slot of fundamental `f` at node `a` (links fill slots
        `0..n−2` first, then flavours grouped by node)."""
        return (self.n_nodes - 1) + sum(self._Nfs[:a]) + f

    # ----- KAlgebra primitives: straight from the flavoured auxiliary -----

    def coefficient_ring(self):
        return self._aux.coefficient_ring()

    def identity(self):
        return self._aux.identity()

    def _label_section_decompose(self, label):
        return self._aux._label_section_decompose(label)

    # ----- RGKAlgebra contract: auxiliary, grading, S_RG = Ψ --------------

    def auxiliary(self):
        return self._aux

    def grading(self):
        """`Γ_RG = Z^L` μ-levels (`L = (n−1) + ΣNf_a`); height = total hyper
        number, one cone generator per slot."""
        L = self._L
        cone = tuple(tuple(1 if j == i else 0 for j in range(L))
                     for i in range(L))
        return Grading(rank=L, deg=lambda lab: tuple(lab[1]),
                       height=(1,) * L, cone_gens=cone)

    # ----- Wilson-character labels per node -------------------------------

    def _wilson_gauge(self, a: int, lam) -> tuple:
        """Pure-U(N_a) gauge label of the Wilson character `χ_λ(v⁽ᵃ⁾)`."""
        key = (a, tuple(lam))
        ch = self._wl.get(key)
        if ch is None:
            N = self._ranks[a]
            full = tuple(lam) + (0,) * (N - len(lam))
            ch = _lowest_charge(L_W(full, N), N)
            self._wl[key] = ch
        return ch

    def _wilson_gauge_inv(self, a: int, nu) -> tuple:
        """Pure-U(N_a) gauge label of `s_ν(v⁻¹) = χ_w`, `w = −rev(ν)` — a
        dominant weight with negative entries; magnetically trivial labels are
        literally `((0,…,0), w)` in the engine convention (NOT a `L_det`
        twist: `det` in the generator registry is the 't Hooft monopole)."""
        key = (a, tuple(nu))
        ch = self._wlinv.get(key)
        if ch is None:
            N = self._ranks[a]
            full = tuple(nu) + (0,) * (N - len(nu))
            w = tuple(-x for x in reversed(full))
            ch = ((0,) * N, w)
            self._wlinv[key] = ch
        return ch

    # ----- folding Ψ across the chain -------------------------------------

    def _edge_content(self, i: int, k: int) -> dict:
        """Link-`i` bifundamental at level `k`: `{(g_a, g_b): Habiro}` over
        per-node gauge labels (node `i` carries `s_λ`, node `i+1` carries
        `s_ν(v⁻¹)`)."""
        key = (i, int(k))
        out = self._edge_cache.get(key)
        if out is None:
            Na, Nb = self._ranks[i], self._ranks[i + 1]
            out = {}
            for (lam, nu), c in bifund_pair_expansion(Na, Nb, k).items():
                ga = self._wilson_gauge(i, lam)
                gb = self._wilson_gauge_inv(i + 1, nu)
                out[(ga, gb)] = c
            self._edge_cache[key] = out
        return out

    def _fuse_node(self, a: int, g, h):
        """Wilson fusion on node `a`: the pure product `g·h` restricted to its
        exact-classical q⁰ part (both labels magnetically trivial)."""
        key = (a, g, h)
        out = self._fuse_cache.get(key)
        if out is None:
            terms = []
            for lab, c in self._pures[a].multiply(g, h).terms.items():
                mult = c._coeffs.get(0, 0)
                if mult:
                    terms.append((lab, int(mult)))
            out = tuple(terms)
            self._fuse_cache[key] = out
        return out

    def _fold_one_node(self, running: dict, a: int, content: dict) -> dict:
        """Multiply the running per-node Wilson product by a single-node factor
        `content = {g: Habiro}` living on node `a`."""
        Z = HabiroElement.zero()
        nxt: dict = {}
        for key, c0 in running.items():
            for g, c in content.items():
                for (lab, mult) in self._fuse_node(a, key[a], g):
                    nk = key[:a] + (lab,) + key[a + 1:]
                    term = c0 * c * mult
                    nxt[nk] = nxt.get(nk, Z) + term
        return {k: v for k, v in nxt.items() if not v.is_zero()}

    def _fold_edge(self, running: dict, i: int, k: int) -> dict:
        """Multiply the running product by link `i` at level `k` (fusing both
        shared nodes)."""
        if k == 0:
            return running
        Z = HabiroElement.zero()
        nxt: dict = {}
        for key, c0 in running.items():
            for (ga, gb), c in self._edge_content(i, k).items():
                for (la, ma) in self._fuse_node(i, key[i], ga):
                    for (lb, mb) in self._fuse_node(i + 1, key[i + 1], gb):
                        nk = key[:i] + (la, lb) + key[i + 2:]
                        term = c0 * c * (ma * mb)
                        nxt[nk] = nxt.get(nk, Z) + term
        return {k2: v for k2, v in nxt.items() if not v.is_zero()}

    def _matter_wilson_content(self, k_vec) -> dict:
        """Wilson content of `[Ψ]_{k_vec}` — the per-slot factors folded across
        the chain.  Returns `{(g₁,…,g_n): HabiroElement}`."""
        n = self.n_nodes
        running = {self._base.identity(): HabiroElement.one()}
        for i in range(n - 1):
            running = self._fold_edge(running, i, int(k_vec[i]))
        for a in range(n):
            for f in range(self._Nfs[a]):
                ki = int(k_vec[self._flavour_slot(a, f)])
                if ki == 0:
                    continue
                content = {self._wilson_gauge(a, lam): c
                           for lam, c in
                           single_hyper_wilson_expansion(self._ranks[a], ki).items()}
                running = self._fold_one_node(running, a, content)
        return running

    def _multi_levels(self, cutoff: int):
        for kv in product(range(cutoff + 1), repeat=self._L):
            if sum(kv) <= cutoff:
                yield kv

    def _s_rg_component(self, p):
        """`[Ψ]_p` — exact matter component at μ-multilevel `p`; `{}` off the
        cone."""
        p = tuple(int(x) for x in p)
        if any(x < 0 for x in p):
            return {}
        return {(g, p): c for g, c in self._matter_wilson_content(p).items()}

    def rg_generator(self, cutoff: int) -> dict:
        """`Ψ` windowed to total hyper number ≤ cutoff (per-node Wilson
        expansion, keyed by auxiliary labels `((lab₁,…,lab_n), k_vec)`)."""
        out: dict = {}
        for k_vec in self._multi_levels(cutoff):
            for g, c in self._matter_wilson_content(k_vec).items():
                out[(g, k_vec)] = c
        return out

    # ------------------------------------------------------------------
    # Chart-evaluated Schur pairing (SPEED-ONLY override; same math as the
    # generic transport — see `un_nf_over_pure_rgflow` for the derivation).
    # Charts are **sums of pure-URQTorus tensors** (one factor per node;
    # scalars absorbed at node 0); the §6b kernel is node-wise:
    #     ⟨⊗_a X_a, ⊗_a Y_a⟩ = ∏_a ⟨X_a, Y_a⟩.
    # ------------------------------------------------------------------

    _CHART_KQ_MARGIN = 16
    _CHART_WINDOW = 2
    _CHART_WINDOW_MAX = 6

    def _chart_of_terms(self, terms, Kq: int):
        """`Σ c(q)·⊗_a urqt(lab_a)` as a list of URQTorus tensors."""
        from urq_torus import URQTorus
        from abelianized_torus import VRational
        out = []
        for lab, c in terms:
            lp = c.expand(Kq) if isinstance(c, HabiroElement) else c
            if lp.is_zero():
                continue
            tens = []
            for j, P in enumerate(self._pures):
                img = P.urqt(lab[j])
                if j == 0:
                    cv = VRational.from_scalar(lp, n=self._ranks[0])
                    img = URQTorus.from_f(
                        {m: (f * cv).simplify()
                         for m, f in img.residuals().items()}, self._ranks[0])
                tens.append(img)
            out.append(tuple(tens))
        return out

    @staticmethod
    def _tensor_mul(s, t):
        return tuple(sa * ta for sa, ta in zip(s, t))

    @staticmethod
    def _tensor_inner(X, Y, K: int):
        """`Σ_{s∈X, t∈Y} ∏_a ⟨s_a, t_a⟩_§6b` (LaurentPoly, kept to `≤ K`).

        Per-node inners can have **negative q-valuation** (monopole-sector
        pairings), so each factor must be computed to depth
        `K + Σ_{b≠a} max(0, −val_b)` before the product is windowed at `K` —
        a uniform-`K` per-node truncation silently corrupts every coefficient
        (caught by the q²/q⁴ drift of `I[E₂,E₂]` with `K`).  The margins are
        learned to a fixed point (a node zero through one window may open at
        a deeper one)."""
        from laurent_poly import LaurentPoly
        acc = LaurentPoly({})
        for s in X:
            for t in Y:
                n = len(s)
                lps = [s[a].inner(t[a], K) for a in range(n)]
                have = [0] * n
                for _ in range(4):
                    vals = []
                    for lp in lps:
                        nz = [e for e, c in lp._coeffs.items() if c]
                        vals.append(min(nz) if nz else 0)
                    needs = [sum(max(0, -vals[b]) for b in range(n) if b != a)
                             for a in range(n)]
                    changed = False
                    for a in range(n):
                        if needs[a] > have[a]:
                            lps[a] = s[a].inner(t[a], K + needs[a])
                            have[a] = needs[a]
                            changed = True
                    if not changed:
                        break
                term = None
                dead = False
                for lp in lps:
                    if not any(lp._coeffs.values()):
                        dead = True
                        break
                    term = lp if term is None else term * lp
                if dead or term is None:
                    continue
                acc = acc + LaurentPoly({e: c for e, c in term._coeffs.items()
                                         if e <= K and c})
        return acc

    def _psi_chart_level(self, k_vec, Kq: int):
        """`chart([Ψ]_{k_vec})` — cached per `(k_vec, Kq)`."""
        cache = self.__dict__.setdefault("_psi_chart_cache", {})
        key = (tuple(k_vec), Kq)
        x = cache.get(key)
        if x is None:
            x = self._chart_of_terms(
                self._matter_wilson_content(k_vec).items(), Kq)
            cache[key] = x
        return x

    def _fs_chart_levels(self, a, top: int, Kq: int) -> dict:
        """`{l: chart([RG(a)·Ψ]_l)}` for μ-levels with `h(l) ≤ top` — exact per
        level (`X_l = Σ_{j+k=l} chart(RG(a)_j)·chart(Ψ_k)`), cached
        incrementally per `(a, Kq)`.  Levels are enumerated from the actual
        `RG(a)` support (which may sit at negative levels after ρ — the
        `un_nf_over_pure_rgflow` blind-spot lesson)."""
        cache = self.__dict__.setdefault("_fs_chart_cache", {})
        key = (a, Kq)
        levels = cache.setdefault(key, {})
        rg_by_level: dict = {}
        for (lab, k_vec), c in self.RG(a).terms.items():
            rg_by_level.setdefault(tuple(k_vec), []).append((lab, c))
        rg_charts = {j: self._chart_of_terms(terms, Kq)
                     for j, terms in rg_by_level.items()}
        wanted = set()
        for j in rg_by_level:
            budget = top - sum(j)
            for k in self._multi_levels(max(budget, -1)):
                wanted.add(tuple(ji + ki for ji, ki in zip(j, k)))
        for l in sorted(wanted):
            if l in levels:
                continue
            acc = []
            for j, chart_j in rg_charts.items():
                k = tuple(li - ji for li, ji in zip(l, j))
                if any(x < 0 for x in k):
                    continue
                for s in chart_j:
                    for t in self._psi_chart_level(k, Kq):
                        acc.append(self._tensor_mul(s, t))
            levels[l] = acc
        return {l: levels[l] for l in sorted(wanted)}

    def _chart_pairing(self, left: dict, right: dict, K: int):
        """`Σ_{k,l} μ^{l−k}·⟨left_k, right_l⟩_§6b` as `RPowerSeries` over the
        flavoured ring."""
        R = self.coefficient_ring()
        acc: dict[int, dict] = {}
        for k, X in left.items():
            for l, Y in right.items():
                lp = self._tensor_inner(X, Y, K)
                if lp.is_zero():
                    continue
                f = tuple(li - ki for li, ki in zip(l, k))
                for e, c in lp._coeffs.items():
                    if e > K or c == 0:
                        continue
                    row = acc.setdefault(e, {})
                    row[f] = row.get(f, 0) + c
        coeffs = {}
        for e, row in acc.items():
            terms = {f: c for f, c in row.items() if c != 0}
            if terms:
                coeffs[e] = RElement(R, terms)
        return RPowerSeries(R, coeffs, K)

    def _chart_pair_stable(self, left_of, right_of, K: int, window):
        """Evaluate at `window` and `window+1`; accept on agreement, else widen
        (honest-fail at `_CHART_WINDOW_MAX`)."""
        W = self._CHART_WINDOW if window is None else int(window)
        prev = None
        while W <= self._CHART_WINDOW_MAX:
            cur = self._chart_pairing(left_of(W), right_of(W), K)
            if prev is not None and prev.coeffs == cur.coeffs:
                return cur
            prev = cur
            W += 1
        raise RuntimeError(
            f"{type(self).__name__}: chart pairing did not stabilise by "
            f"window {self._CHART_WINDOW_MAX} (K={K}); widen `window=` or "
            f"fall back to the generic path (super().trace/inner_product).")

    def trace(self, a, K: int = 20, window=None):
        """`Tr(L_a)` — chart-evaluated FS pairing (speed-only override)."""
        Kq = K + self._CHART_KQ_MARGIN
        h_a = sum(self.grading().deg(a))
        psi = lambda W: {k: self._psi_chart_level(k, Kq)
                         for k in self._multi_levels(W)}
        fs = lambda W: self._fs_chart_levels(a, h_a + W, Kq)
        return self._chart_pair_stable(psi, fs, K, window)

    def inner_product(self, a, b, K: int = 20, window=None):
        """`I_{a,b} = Tr(ρ(L_a)·L_b)` — chart-evaluated FS pairing (speed-only
        override; generic path via `super().inner_product`)."""
        Kq = K + self._CHART_KQ_MARGIN
        h_a = sum(self.grading().deg(a))
        h_b = sum(self.grading().deg(b))
        fs_a = lambda W: self._fs_chart_levels(a, h_a + W, Kq)
        fs_b = lambda W: self._fs_chart_levels(b, h_b + W, Kq)
        return self._chart_pair_stable(fs_a, fs_b, K, window)


# ---------------------------------------------------------------------------
# Demonstration.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for ranks in [(2, 1), (2, 2)]:
        A = QuiverOverPure(ranks)
        tag = "×".join(f"U({N})" for N in ranks)
        print(f"==============  {tag} + bifundamental over ⊗ pure  ==============")
        S = A.rg_generator(2)
        print(f"  Γ_RG rank {A.grading().rank}; S_RG (Σ level ≤ 2): {len(S)} terms")
        for lab, c in sorted(S.items(), key=lambda t: (sum(t[0][1]), str(t[0]))):
            print(f"    {lab}:  {c}")
        print()
