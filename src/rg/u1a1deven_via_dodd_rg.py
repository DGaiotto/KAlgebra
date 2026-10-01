"""
u1a1deven_via_dodd_rg.py
========================

`U1A1DevenViaDoddRG(k)` — a **fast** RG oracle for the u(1)-gauged
`[A_1, D_{2k+2}]` theory (the `U1A1DevenConeKAlgebra` target), built — *just as
for `u1A1Aodd -> A1Aeven`* — over the **ungauged odd-D
sibling** instead of the A-type `A1A2k ⊗ QT ⊗ SU(2)` with a doublet `S_RG`:

    auxiliary = A1DoddConeKAlg(k-1) ⊗ QT(Z²)          # A1Dodd(k-1) = [A_1, D_{2k+1}]
    S_RG      = E_𝖖(X_{0,1} · L)                       # SINGLE letter, L a short chord

The SU(2) flavour is **intrinsic** to `A1DoddConeKAlg` (coefficient ring
`SU2ZPlusRing`), so this drops both the `⊗ SU(2)` factor *and* the doublet
Schur-peel of the legacy `U1A1DevenRGKAlgebra` (whose `S_RG = E(μX01L)E(μ⁻¹X01L)`
is the dominant cost).  With a single-letter `S_RG` over the fast closed-form
A1Dodd cone, the RG-transport trace is far cheaper — the point being a *fast
ground truth* for the k≥2 matter trace (where the legacy oracle is ~minutes/seed).

Structure mirrors `u1a1aodd_gauged_rg.U1A1AoddGaugedRG` exactly:
  * labels `(dodd_label, (c0, c1))` — `dodd_label = (word, κ)` an A1Dodd label,
    `(c0, c1)` the QT(Z²) charge; `c1` the **dressed** leg (Γ_RG = Z grades by it),
    `c0` the **free gauge** leg (the physical X_{0,1} tower of the gauged theory).
  * `S_RG = E_𝖖(X_{(0,1)} · L)`: degree-`m` part `(L^m, (0, m))` with Habiro `c_m`.
  * apex = identity; Γ_RG = Z = `c1`.

`L` is the **length-`k` DOUBLET** chord `(k, 1, i)` of `A1Dodd(k-1)` — the
*longest* (length k) `p=1` (SU(2) χ₁) chord; *not* a `p=0` singlet and *not* a
shorter doublet (see STATUS).  The pairing `dodd_index = k-1` (gauged `D_{2k+2}`
over `D_{2k+1}`) and `L` are *validated by matching* the literature-verified
vacuum index `deven_gauged_xn_qn(k, 0)` exactly (k=1,2,3).

STATUS (2026-06-24).
* **Flow validated, all k, all K**: with `L = (k, 1, i)` the **length-k doublet**
  chord, `Tr(1)` reproduces the literature-verified vacuum index
  `deven_gauged_xn_qn(k, 0)` **exactly** — k=1 (q¹⁴), k=2, k=3 — and is
  cross-validated against BOTH the legacy `U1A1DevenRGKAlgebra` and the standalone
  `U1A1DevenConeKAlgebra` (three-way bench: all agree).
* **`L` = length-k doublet chord `(k, 1, i)` — THE fix (2026-06-24).**  Two
  conditions, both essential (each found the hard way):
  1. **DOUBLET (`p=1`).** L's A1Dodd trace must carry half-integer spins
     (χ₁,χ₃,…) so the single-letter `E_𝖖(X01·L)` reproduces the legacy *doublet*
     `S_RG = E(μX01L)E(μ⁻¹X01L)` folding intrinsically (μ^{±1} Weyl-fold carried
     by L's own SU(2)).  A `p=0` SINGLET (χ₀-only) misses it → `Tr(1)` right only
     to q² (the old default `(1,0,0)`; an earlier note misdiagnosed this stable
     wrongness as "windowing drift" — it is a wrong-L artifact).
  2. **LENGTH k.** The chord length scales with k (the deven theory's shortest
     chord = the *survivor's longest*).  A shorter doublet OVER-counts matter
     from q⁶: at k=2 the length-1 doublet is wrong from q⁶, at k=3 lengths 1 & 2
     are wrong from q⁶, only length-k is exact — matching the author's "chord length
     grows with k (same or longer by 1)".
  Any third index `i` works.  All correct cases are FAST (~0.7s for `Tr(1)` to
  q⁸; the correct flow converges in the first window — no plateau).
* **Speed (three-way bench, k=2, Tr(1)→q⁸):** NEW ≈0.8s vs CONE ≈20s (cold
  build) vs LEGACY ≈25s.  Single-letter `S_RG` over the closed-form A1Dodd cone;
  coefficients truncated to q≤K (`_trunc`) — the deep `S_RG` Habiro tower has
  QT-neutral coefficients out to ~q²⁶⁴ that can't reach a K-order trace; not
  truncating made `aux.trace_element`'s SU(2) arithmetic chew ≈145s (now ≈0.01s).
* **Use.** A fast, flow-correct fast oracle / ground truth for the k≥2 matter
  trace (legacy ~minutes/seed there).
* **X_{0,1} leg RESOLVED (2026-06-25) — `u1a1deven_cone_new_iso` (in the source repository's archive
  since 2026-09-23, when the cone was rebuilt on this flow; since 2026-09-24 the
  cone `U1A1DevenConeKAlgebra` is the curve frame derived from this flow's RG
  image, its labels in closed-form bijection with this flow's).**  The
  cross-oracle `KAlgebraIso  U1A1DevenConeKAlgebra(k) ≅ U1A1DevenViaDoddRG(k)`
  pins the labels (k=1: 256/256 mult, 16/16 ρ).  The **cone's `X_{0,1}`**
  (electric, the clean Laurent torus it mods out) is the **`c1=(0,1)` leg here**
  (generator `((( ),0),(0,1))`); `c0` is the bounded **magnetic** `X_{1,0}`
  charge (∈ {−1,0,1}, the cone's `c1_mag`).  So despite the "free gauge / X_{0,1}
  tower" wording above for `c0`, the *cone-`X_{0,1}`* Laurent direction is `c1`.
  Both oracles are **Z-form** (SU(2) in the A1Dodd word / κ-label, χ₀
  coefficients; the cone was R-form until 2026-09-24, bridged by
  `r_label_decompose`, and is Z-form since).  The
  bijection is non-canonical (RG fixes the basis up to relabeling): discovered by
  ρ-orbit + magnetic charge + Plücker fan-out, fixed by ρ-intertwining +
  multiplicativity, with a per-orbit `X_{0,1}` dressing (`ρ(X01)=X01⁻¹` both
  sides).
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
from a1dodd_kalg import A1DoddConeKAlg


def _e_q_coeff(m: int) -> HabiroElement:
    """`c_m = (−q)^m / (q²;q²)_m` — the m-th coeff of `E_𝖖(X)` for a
    self-pairing-free generator (⟨X01·L, X01·L⟩ = 0)."""
    num = LaurentPoly({m: (-1) ** m})
    denom = {j: 1 for j in range(1, m + 1)}
    return HabiroElement(num, denom)


class U1A1DevenViaDoddRG(RGKAlgebra):
    """u(1)-gauged `[A_1, D_{2k+2}]` as an `RGKAlgebra` over
    `A1DoddConeKAlg(k-1) ⊗ QT(Z²)`, dressing the electric leg by a short chord
    `L` with `S_RG = E_𝖖(X_{(0,1)} · L)` (single letter)."""

    def __init__(self, k: int, chord=None):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self._dodd_index = k - 1                       # A1Dodd(k-1) = [A_1, D_{2k+1}]
        self._surv = A1DoddConeKAlg(self._dodd_index)
        self._qt = QuantumTorusKAlg([[0, 1], [-1, 0]])
        self._aux = TensorKAlgebra(self._surv, self._qt)
        # L MUST be a LENGTH-k DOUBLET chord `(k, 1, i)` of A1Dodd(k-1) — the
        # *longest* (= length k) doublet (p=1, SU(2) χ₁) chord (verified k=1,2,3,
        # 2026-06-24; any i works).  TWO conditions, both essential:
        #   * DOUBLET (p=1): its A1Dodd trace carries half-integer spins
        #     (χ₁,χ₃,…), so single-letter E_𝖖(X01·L) reproduces the legacy
        #     doublet S_RG = E(μX01L)E(μ⁻¹X01L) folding intrinsically.  A p=0
        #     singlet (χ₀-only) misses it → Tr(1) right only to q².
        #   * LENGTH k: the chord length scales with k.  A *shorter* doublet
        #     over-counts matter from q^{2k+2} (k=2: len-1 wrong from q⁶; k=3:
        #     len-1,2 wrong from q⁶) — exactly the author's "chord length grows
        #     with k" (the deven theory's shortest chord = the survivor's
        #     longest).  Default `(k, 1, 0)`.
        self._L = (k, 1, 0) if chord is None else tuple(chord)

    # ----- RGKAlgebra contract -------------------------------------------

    def coefficient_ring(self):
        return self._aux.coefficient_ring()

    def identity(self):
        return self._aux.identity()

    def auxiliary(self):
        return self._aux

    def grading(self) -> Grading:
        # Γ_RG = Z = the DRESSED QT leg (0,1) charge c1; the (1,0) leg c0 is the
        # free gauge direction (the physical X_{0,1} tower).  Cone Z_{>=0}.
        return Grading(rank=1, deg=lambda lbl: (lbl[1][1],), height=(1,),
                       cone_gens=((1,),))

    def apex(self, a):
        """Identity apex: UV labels coincide with auxiliary labels."""
        return (a[0], tuple(a[1]))

    def _dodd_label_pow(self, m: int):
        """`L^m` as an A1Dodd label `(word, κ=0)`: the single chord `L` to
        power `m` (`word = ((L, m),)`)."""
        if m == 0:
            return self._surv.identity()
        return (((self._L, m),), 0)

    def _s_rg_component(self, p):
        """`[S_RG]_{(m,)}` — `S_RG = E_𝖖(X_{(0,1)}·L)`, so the degree-`m` part
        (in the dressed leg (0,1)) is `(L^m, (0, m))` with Habiro `c_m`;
        degree 0 = auxiliary identity, negative degree empty."""
        (m,) = p
        if m < 0:
            return {}
        if m == 0:
            return {self._aux.identity(): _e_q_coeff(0)}
        return {(self._dodd_label_pow(m), (0, m)): _e_q_coeff(m)}

    def rg_generator(self, cutoff: int) -> dict:
        out: dict = {}
        for m in range(cutoff):
            out.update(self._s_rg_component((m,)))
        return out

    def _section_split(self, label):
        # nested 2-tuple QT charge: disable the flat-vector flavour-shift cache.
        return tuple(label), None

    # ----- adaptive-window trace (mirrors u1a1aodd_gauged_rg) -------------

    def trace(self, a, K: int = 20):
        """`Tr_UV(L_a) = Tr_aux(ρ(S_RG)·RG(a)·S_RG)`, S_RG windowed with
        multi-window stability (≥3 consecutive agreeing windows; `win_max=6K+48`)
        — the linear-Habiro-valuation sizing from the aodd oracle."""
        cache = self.__dict__.setdefault("_adaptive_trace_cache", {})
        ck = (a, K)
        if ck in cache:
            return cache[ck]
        aux = self.auxiliary()
        rg_a = self.RG(a)
        prev = result = None
        agree_run = 0
        win = max(self._rg_cutoff(), 2)
        win_max = 6 * K + 48
        from kalgebra import Element
        from laurent_poly import LaurentPoly
        from zplus_ring import RLaurent

        def _trunc(c):
            """Drop coefficient q-powers > K: the deep `S_RG` Habiro tower gives
            coefficients out to ~q^{(win·levels)} (e.g. q²⁶⁴), but a K-order trace
            (trace q-series start at q⁰) only sees coefficient q-powers ≤ K.
            Without this, `trace_element`'s SU(2) RPowerSeries arithmetic chews
            through the huge coefficients — 144s vs 0.01s at k=1 K=8."""
            co = getattr(c, "coeffs", None)
            if co is not None:                            # RLaurent (SU(2))
                return RLaurent(c.R, {q: v for q, v in co.items() if q <= K})
            return LaurentPoly({q: v for q, v in c._coeffs.items() if q <= K})

        while win <= win_max:
            try:
                s_rg = self._s_rg_as_aux_element(win, K + win)
                rho_s = aux.rho_element(s_rg)
                prod = aux.multiply_elements(rho_s, aux.multiply_elements(rg_a, s_rg))
                # QT trace = δ_{(c0,c1),0}; keep only QT-neutral terms (the legacy
                # oracle's filter) AND truncate their coefficients to q≤K (the deep
                # S_RG tower's high-q coefficients can't reach a K-order trace).
                neutral = Element({lbl: _trunc(c) for lbl, c in prod.terms.items()
                                   if lbl[1] == (0, 0)})
                tr = aux.trace_element(neutral, K)
            except RuntimeError:
                break
            if prev is not None and tr.coeffs == prev.coeffs:
                agree_run += 1
                if agree_run >= 3:
                    result = tr
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
    T = U1A1DevenViaDoddRG(1)
    print("U1A1DevenViaDoddRG(1): aux = A1Dodd(0)=D3 ⊗ QT, L =", T._L)
    print("coeff ring =", T.coefficient_ring())
    g = ((((1, 0, 1), 1),), 0)
    print("RG(g@(0,0)) =", {l: str(c) for l, c in T.RG((g, (0, 0))).terms.items()})
