"""un_nf_over_pure_rgflow — U(N)+N_f as an `RGKAlgebra` wrapping pure U(N).

The recipe (the standard one, used across the repo — cf.
`su2_nf_over_pure_rgflow`, `u1_nf_rgflow`):

  1. take **pure U(N)** = `pure_un_kalgebra.PureUNKAlgebra`;
  2. promote it to the **U(N_f)-flavoured** K-algebra the standard way —
     `add_flavour(AbelianZPlusRing(N_f))`;
  3. wrap it in an `RGKAlgebra` with the matter spectrum generator
     `S_RG = Ψ = ∏_{i=1}^{N_f} ∏_{j=1}^N E_𝔮(μ_i v_j)` as the only extra datum.

`Ψ` is expanded on the pure-U(N) **Wilson-line characters** `χ_λ(v)` (its
graded components `_s_rg_component` / `rg_generator`).  The generic
`RGKAlgebra` machinery then derives the entire algebra — `RG`, `multiply`,
`trace`, `inner_product`, ρ — from the auxiliary's own multiplication and
trace.  The only overrides are **speed-only**: `trace` / `inner_product`
evaluate the same FS pairing chart-side (per-flavour-level `URQTorus`
images through the §6b kernel — see the "Chart-evaluated Schur pairing"
section below; the generic transport stays reachable via `super()` and is
the cross-validation reference).  `UNNfOverPure(N, N_f)` is the general
class; `UNNf1OverPure(N)` is the N_f=1 case.
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

from rgkalgebra import RGKAlgebra
from grading import Grading
from habiro import HabiroElement
from zplus_ring import AbelianZPlusRing, RElement, RPowerSeries

from pure_un_kalgebra import PureUNKAlgebra
from pure_un_construct import _lowest_charge
from pure_un_canonical import L_W
from pure_un_chart_engine import _kostka, _partitions_n_parts


# ---------------------------------------------------------------------------
# E_𝔮 coefficients and the matter spectrum Ψ = ∏_{i,j} E_𝔮(μ_i v_j),
# expanded on the pure-U(N) Wilson-line characters χ_λ(v).
# ---------------------------------------------------------------------------


def _a(n: int) -> HabiroElement:
    """`a_n = [E_𝔮(x)]_n = (-1)^n q^n / (q^2;q^2)_n`, exact (0 for n < 0)."""
    if n < 0:
        return HabiroElement.zero()
    return HabiroElement.nahm_term((-1) ** n, n, [n])


def _partitions_le_parts(k: int, N: int):
    out = set()
    for p in _partitions_n_parts(k, N):
        out.add(tuple(x for x in p if x > 0))
    return sorted(out, reverse=True)


def _inv_kostka(k: int, N: int):
    """`(parts, Kinv)` for partitions of `k` with ≤ N parts (inverse Kostka)."""
    parts = _partitions_le_parts(k, N)
    n = len(parts)
    K = [[Fraction(_kostka(parts[i], parts[j])) for j in range(n)]
         for i in range(n)]
    Inv = [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = next(r for r in range(col, n) if K[r][col] != 0)
        K[col], K[piv] = K[piv], K[col]
        Inv[col], Inv[piv] = Inv[piv], Inv[col]
        pv = K[col][col]
        K[col] = [x / pv for x in K[col]]
        Inv[col] = [x / pv for x in Inv[col]]
        for r in range(n):
            if r != col and K[r][col] != 0:
                f = K[r][col]
                K[r] = [a - f * b for a, b in zip(K[r], K[col])]
                Inv[r] = [a - f * b for a, b in zip(Inv[r], Inv[col])]
    return parts, Inv


def single_hyper_wilson_expansion(N: int, k: int):
    """One fundamental hyper at flavour level `k`: `[∏_j E_𝔮(μ v_j)]_{μ^k}` as
    `{λ: c_{k,λ}}` over Wilson characters `χ_λ`, `c_{k,λ} = Σ_μ (K^{-1})_{μλ}
    ∏ a_{μ_i}` (exact Habiro)."""
    parts, Inv = _inv_kostka(k, N)
    aprod = []
    for mu in parts:
        acc = HabiroElement.one()
        for part in mu:
            acc = acc * _a(part)
        aprod.append(acc)
    out = {}
    for li, lam in enumerate(parts):
        c = HabiroElement.zero()
        for mj in range(len(parts)):
            coeff = Inv[mj][li]
            if coeff != 0:
                c = c + aprod[mj] * int(coeff.numerator)
        if not c.is_zero():
            out[lam] = c
    return out


def _pure_un_dressed_rays(N: int, dress: int):
    """Dressed E/F monopole rays + det^±1 — a registry generous enough that the
    Wilson sector the matter flow exercises is closed."""
    rays = []
    for k in range(1, N):
        for f in product(range(-dress, dress + 1), repeat=N):
            rays.append(('E', k, f))
            rays.append(('F', k, f))
    rays += [('det', 1), ('det', -1)]
    return rays


# ---------------------------------------------------------------------------
# The RGKAlgebra: pure U(N) ⊗ U(N_f)-flavour + S_RG = Ψ.  Nothing else.
# ---------------------------------------------------------------------------


class UNNfOverPure(RGKAlgebra):
    """U(N) + N_f fundamental hypers, built by promoting pure U(N) to the
    flavoured K-algebra (`add_flavour`) and supplying `S_RG = Ψ`.  The generic
    `RGKAlgebra` derives `RG` / `multiply` / `trace` / `inner_product`."""

    def __init__(self, N: int, Nf: int = 1, dress: int = 1, max_len: int = 2,
                 wilson_deg: int = 4, K: int = 8) -> None:
        self._N = int(N)
        self._Nf = int(Nf)
        self._pure = PureUNKAlgebra(self._N, _pure_un_dressed_rays(self._N, dress),
                                    max_len=max_len, K=K, wilson_deg=wilson_deg)
        self._aux = self._pure.add_flavour(AbelianZPlusRing(rank=self._Nf))
        self._wl: dict[tuple, tuple] = {}

    @property
    def N(self) -> int:
        return self._N

    @property
    def Nf(self) -> int:
        return self._Nf

    def pure(self) -> PureUNKAlgebra:
        return self._pure

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
        """`Γ_RG = Z^{N_f}` flavour levels (height = total flavour number; one
        cone generator per flavour)."""
        Nf = self._Nf
        cone = tuple(tuple(1 if j == i else 0 for j in range(Nf))
                     for i in range(Nf))
        return Grading(rank=Nf, deg=lambda lab: tuple(lab[1]),
                       height=(1,) * Nf, cone_gens=cone)

    def _wilson_gauge(self, lam) -> tuple:
        """Pure-U(N) gauge charge of the Wilson character `χ_λ` (memoized)."""
        key = tuple(lam)
        ch = self._wl.get(key)
        if ch is None:
            full = key + (0,) * (self._N - len(key))
            ch = _lowest_charge(L_W(full, self._N), self._N)
            self._wl[key] = ch
        return ch

    def _matter_wilson_content(self, k_vec):
        """Wilson content of `[Ψ]_{k_vec} = ∏_i [Ψ_i]_{k_i}` — the LR fusion of
        the per-flavour single-hyper Wilson expansions (Schur×Schur via the pure
        Wilson multiply).  Returns `{gauge_wilson_label: HabiroElement}`."""
        N = self._N
        content = {self._wilson_gauge(()): HabiroElement.one()}
        for ki in k_vec:
            exp = single_hyper_wilson_expansion(N, int(ki))
            nxt: dict = {}
            for g, cg in content.items():
                for lam, cl in exp.items():
                    lg = self._wilson_gauge(lam)
                    for mug, lr in self._pure.multiply(g, lg).terms.items():
                        mult = lr._coeffs.get(0, 0)        # Wilson LR mult (q⁰)
                        if mult:
                            term = cg * cl * int(mult)
                            nxt[mug] = (nxt.get(mug, HabiroElement.zero()) + term)
            content = {g: c for g, c in nxt.items() if not c.is_zero()}
        return content

    def _multi_levels(self, cutoff: int):
        for kv in product(range(cutoff + 1), repeat=self._Nf):
            if sum(kv) <= cutoff:
                yield kv

    def _s_rg_component(self, p):
        """`[Ψ]_p` — exact matter component at flavour multilevel `p`; `{}` off
        the cone."""
        p = tuple(int(x) for x in p)
        if any(x < 0 for x in p):
            return {}
        return {(g, p): c for g, c in self._matter_wilson_content(p).items()}

    def rg_generator(self, cutoff: int) -> dict:
        """`Ψ` windowed to total flavour number ≤ cutoff (the `χ_λ` Wilson-line
        expansion, keyed by auxiliary labels `((m,e), k_vec)`)."""
        out: dict = {}
        for k_vec in self._multi_levels(cutoff):
            for g, c in self._matter_wilson_content(k_vec).items():
                out[(g, k_vec)] = c
        return out

    # ------------------------------------------------------------------
    # Chart-evaluated Schur pairing (SPEED-ONLY override; same math).
    #
    # The generic `trace` / `inner_product` transport through the auxiliary
    # by multiplying FS objects in the *canonical basis* — every cross term
    # is a flavoured pure-U(N) `multiply` (closed-form-engine products, the
    # expensive part).  But the auxiliary's trace only ever reads the chart
    # (`PureUNKAlgebra.trace`/`inner_product` evaluate on `URQTorus` images),
    # and the chart is faithful for trace purposes, so the whole pairing can
    # be evaluated chart-side:
    #
    #   FS_a := RG(a)·Ψ = Σ_l μ^l · X_l   (X_l pure chart objects, exact per
    #                                       level by flavour conservation:
    #                                       X_l = Σ_{j+k=l} chart(RG(a)_j)·chart(Ψ_k))
    #   I_{a,b} = Tr_aux(ρ_aux(FS_a)·FS_b)
    #           = Σ_{k,l} μ^{l−k} · ⟨X_k, Y_l⟩_§6b        (URQTorus.inner)
    #   Tr(a)   = the same sum with the left factor S_RG = Ψ itself.
    #
    # Windows: flavour height ≤ base + `window` with a two-window stability
    # check (the #312 pattern; a residual marching with the window is the
    # truncation signature), and Habiro coefficients
    # q-expanded at `Kq = K + _CHART_KQ_MARGIN` (their q-lead grows with the
    # flavour level, so the margin dominates the down-shifts seen in
    # practice; the suite cross-validates against the independently-windowed
    # generic path).  `verify_inner_product_consistent` /
    # `verify_trace_pairing_faces` remain the contract-level cross-checks.
    # ------------------------------------------------------------------

    _CHART_KQ_MARGIN = 16
    _CHART_WINDOW = 2
    _CHART_WINDOW_MAX = 6

    def _chart_of_terms(self, terms, Kq: int):
        """`Σ c(q)·urqt(lab)` as one `URQTorus` (terms: `(pure_label, coeff)`
        with `coeff` a `LaurentPoly` or exact `HabiroElement`, expanded at `Kq`)."""
        from urq_torus import URQTorus
        from abelianized_torus import VRational
        N = self._N
        acc = URQTorus.zero(N)
        for lab, c in terms:
            lp = c.expand(Kq) if isinstance(c, HabiroElement) else c
            if lp.is_zero():
                continue
            cv = VRational.from_scalar(lp, n=N)
            img = self._pure.urqt(lab)
            acc = acc + URQTorus.from_f(
                {m: (f * cv).simplify() for m, f in img.residuals().items()}, N)
        return acc

    def _psi_chart_level(self, k_vec, Kq: int):
        """`chart([Ψ]_{k_vec})` — cached per `(k_vec, Kq)`."""
        cache = self.__dict__.setdefault("_psi_chart_cache", {})
        key = (tuple(k_vec), Kq)
        x = cache.get(key)
        if x is None:
            x = self._chart_of_terms(self._matter_wilson_content(k_vec).items(), Kq)
            cache[key] = x
        return x

    def _fs_chart_levels(self, a, top: int, Kq: int) -> dict:
        """`{l: chart([RG(a)·Ψ]_l)}` for flavour levels with `h(l) ≤ top` —
        exact per level (`X_l = Σ_{j+k=l} chart(RG(a)_j)·chart(Ψ_k)`), cached
        incrementally per `(a, Kq)`.

        Levels are enumerated from the **actual `RG(a)` support**: for each
        support level `j`, the Ψ-cone walk contributes `l = j + k` with
        `h(k) ≤ top − h(j)`.  (`RG(a)` levels can be negative — e.g. `μ⁻¹`
        sits at level −1 — and the cone-only enumeration that preceded this
        silently dropped those blocks: the level-(−1) FS block of `Tr(μ⁻¹)`
        carried the q⁰ term, and the W-stability loop cannot see a blind
        spot shared by every window.  Caught by the T7 iso battery's
        trace-equivariance leg against BPS.)"""
        from urq_torus import URQTorus
        cache = self.__dict__.setdefault("_fs_chart_cache", {})
        key = (tuple(a), Kq)
        levels = cache.setdefault(key, {})
        rg_by_level: dict = {}
        for (lab, k_vec), c in self.RG(a).terms.items():
            rg_by_level.setdefault(tuple(k_vec), []).append((lab, c))
        wanted = set()
        for j in rg_by_level:
            budget = top - sum(j)
            for k in self._multi_levels(max(budget, -1)):
                wanted.add(tuple(ji + ki for ji, ki in zip(j, k)))
        for l in sorted(wanted):
            if l in levels:
                continue
            acc = URQTorus.zero(self._N)
            for j, terms in rg_by_level.items():
                k = tuple(li - ji for li, ji in zip(l, j))
                if any(x < 0 for x in k):
                    continue
                rg_chart = self._chart_of_terms(terms, Kq)
                acc = acc + rg_chart * self._psi_chart_level(k, Kq)
            levels[l] = acc
        return {l: levels[l] for l in sorted(wanted)}

    def _chart_pairing(self, left: dict, right: dict, K: int):
        """`Σ_{k,l} μ^{l−k}·⟨left_k, right_l⟩_§6b` as `RPowerSeries` over the
        flavour ring (negative-q content, if any, is kept — the strengthened
        orthonormality verifier is the judge of it)."""
        R = self.coefficient_ring()
        acc: dict[int, dict] = {}
        for k, X in left.items():
            for l, Y in right.items():
                lp = X.inner(Y, K)
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
        """Evaluate the chart pairing at `window` and `window+1`; accept on
        agreement, else widen (honest-fail at `_CHART_WINDOW_MAX`)."""
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
            f"fall back to the generic path (super().trace/inner_product)."
        )

    def trace(self, a, K: int = 20, window=None):
        """`Tr(L_a)` — chart-evaluated FS pairing (speed-only override of the
        generic RG transport; same math, §6b kernel)."""
        Kq = K + self._CHART_KQ_MARGIN
        h_a = sum(self.grading().deg(a))
        psi = lambda W: {k: self._psi_chart_level(k, Kq)
                         for k in self._multi_levels(W)}
        fs = lambda W: self._fs_chart_levels(a, h_a + W, Kq)
        return self._chart_pair_stable(psi, fs, K, window)

    def inner_product(self, a, b, K: int = 20, window=None):
        """`I_{a,b} = Tr(ρ(L_a)·L_b)` — chart-evaluated FS pairing (speed-only
        override; the generic path remains via `super().inner_product`)."""
        Kq = K + self._CHART_KQ_MARGIN
        h_a = sum(self.grading().deg(a))
        h_b = sum(self.grading().deg(b))
        fs_a = lambda W: self._fs_chart_levels(a, h_a + W, Kq)
        fs_b = lambda W: self._fs_chart_levels(b, h_b + W, Kq)
        return self._chart_pair_stable(fs_a, fs_b, K, window)


class UNNf1OverPure(UNNfOverPure):
    """U(N) + N_f=1 — the single-hyper special case of `UNNfOverPure`."""

    def __init__(self, N: int, dress: int = 1, max_len: int = 2,
                 wilson_deg: int = 4, K: int = 8) -> None:
        super().__init__(N, 1, dress=dress, max_len=max_len,
                         wilson_deg=wilson_deg, K=K)


# ---------------------------------------------------------------------------
# Demonstration.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    A = UNNf1OverPure(2)
    print("==============  U(2) + N_f=1  over pure U(2)  ==============")
    print("  recipe: pure U(2) → add_flavour → RGKAlgebra(S_RG = Ψ); generic API\n")
    print("  S_RG = Ψ on Wilson lines (level ≤ 3):")
    for lab, c in sorted(A.rg_generator(3).items(), key=lambda t: (sum(t[0][1]), t[0])):
        print(f"    {lab}:  {c}")
    print("\n  identity:", A.identity())
    print("  Tr(identity, 6) =", A.trace(A.identity(), 6))
    # Wilson sector (gauge): χ_□ · χ_□
    c1 = (A._wilson_gauge((1,)), (0,))
    print("  χ_□ · χ_□ =", {k: str(v) for k, v in A.multiply(c1, c1).terms.items()})
    print("  I[χ_□, χ_□](6) =", A.inner_product(c1, c1, 6))
