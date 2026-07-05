"""`SU2Nf1Abe` — SU(2)+N_f=1 as a genuine **abelianized** realization on the
**rational quantum torus** (the URQTorus `abe` leg, 2026-06-13).

This is the genuine URQTorus realization the `bps-rform` leg stood in for:
it lives on the **enriched rational quantum torus** (`MatterURQTorus`,
the `v`-variable cone-atom presentation with the matter rungs) — *not*
the BPS integer-charge chart.  It is built over `UNNfKAlgebra(2, 1)`
(the U(2)+N_f=1 `AbeKAlgebra` on the rational quantum torus), so "Abe" in
the name is the URQTorus sense (user ruling: that name is reserved for
rational-quantum-torus realizations).

The flavour, correctly (no Z₂).  The earlier U(2)-trace-zero attempt read
the **gauge** U(1) centre (`λ₁+λ₂`) as the flavour, which is
Z₂-entangled with the SU(2) spinor charge — wrong.  Here the flavour μ is
the **matter level** `k` of the rational quantum torus (the hyper's
U(1)_F, a coordinate genuinely *separate* from the gauge centre), and the
gauge centre is collapsed (SU(2) reading).  Concretely a torus element is
attributed to U(2)+N_f=1 lines `((m⃗, λ), k)` and mapped to SU(2)

    (m⃗, λ, k)  ↦  (m = m⃗[0],  e = λ₁ − λ₂,  μ = μ^k),

dropping the gauge centre `λ₁+λ₂`.  Verified: `H0·W1 = q·H1 + q⁻¹·H-1 +
μ·1` with the meson at matter-level 1 and the dressed monopoles / Wilsons
at level 0 (μ⁰) — exactly cone/BPS, no Z₂.

Labels `(m, e)` — SU(2) magnetic `m ≥ 0`, electric `e` (Weyl-folded); the
flavour μ lives in the coefficient ring `R = AbelianZPlusRing(1)` (the
matter level), so generators carry no flavour charge.  `multiply` / `rho`
run on the rational quantum torus (`MatterURQTorus` product / GTwist-ρ,
keeping the matter levels) and are read back to SU(2) labels by the
level-ascending attribution.

The **trace is the native SU(2)+N_f=1 Schur trace** — the **rank-1 SU(2)**
Schur-measure residue, NOT the inherited U(2) `MatterURQTorus.trace`.  The
pure-U(2) residue (`trace_v0`) carries the prefactor `(q²;q²)_∞^{2N} =
(q²;q²)_∞⁴` (rank 2 = the SU(2) Cartan factor × the decoupled U(1)-photon
factor); SU(2) (rank 1) has **one fewer** `(q²;q²)_∞²` factor — the
ungauged central-U(1) photon (user, 2026-06-13) — so the residue is
`trace_v0 / (q²;q²)_∞²`.  The central U(1) is kept as the flavour fugacity:
a content×matter-factor term of matter level `lk+n` surfaces at `μ^{lk+n}`,
with its v₀v₁-degree shifted to 0 before the v-residue (an odd central
charge is odd under the SU(2) Cartan and integrates to 0).  Certified `==`
the cone / BPS SU(2)+N_f=1 trace.  (The matter factor keeps exact Habiro
`nahm_term` coefficients, expanded only at the boundary, with a
window-stability guard.)
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import KAlgebra, Element
from laurent_poly import LaurentPoly
from zplus_ring import AbelianZPlusRing, RElement, RLaurent, RPowerSeries


__all__ = ["SU2Nf1Abe", "su2_fold"]


def su2_fold(m: int, e: int) -> tuple:
    """Fold `(m, e)` into the SU(2) Weyl chamber: `(m,e) ↦ (−m,−e)`; pick
    `m ≥ 0`, and at `m = 0` pick `e ≥ 0`."""
    if m < 0:
        m, e = -m, -e
    if m == 0 and e < 0:
        e = -e
    return (m, e)


class SU2Nf1Abe(KAlgebra):
    """SU(2)+N_f=1 on the rational quantum torus (via `UNNfKAlgebra(2,1)`),
    flavour = the matter level (no Z₂).  See the module docstring."""

    def __init__(self) -> None:
        from un_nf_kalgebra import UNNfKAlgebra
        self._U = UNNfKAlgebra(2, 1)
        self._R = AbelianZPlusRing(rank=1)

    # ----- KAlgebra basics -----------------------------------------------

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return (0, 0)

    # ----- SU(2) ↔ U(2)+N_f=1 lift / read --------------------------------

    @staticmethod
    def _lift(label):
        """SU(2) `(m, e)` → U(2)+N_f=1 gauge label `((m,−m), λ)`, λ the
        det-representative with `λ₁−λ₂ = e` (central collapsed)."""
        m, e = label
        lam = (e, 0) if e >= 0 else (0, -e)
        return ((m, -m), lam)

    def _chart(self, label):
        """The canonical line as a `MatterWRQTorus` (the U(2)+N_f=1
        rational-quantum-torus chart of the lift) — the WRQ substrate, so
        multiply/ρ/`_read` route through `UNNfKAlgebra._attribute` (WRQ)."""
        return self._U.chart(self._lift(label))

    def _urq_chart(self, label):
        """The URQ (`VRational`-engine) chart of the lift — the native SU(2)
        Schur-trace path below reads the `VRational` residual internals
        (`_matter_factor_levels` / `trace_v0`), which are engine-specific and
        do not consume the WRQ `TorusRational` residuals."""
        return self._U._urq_chart(self._lift(label))

    def _read(self, torus) -> Element:
        """Attribute a `MatterURQTorus` element to SU(2) `(m, e)` lines with
        the matter level folded onto μ ∈ R, collapsing the gauge centre."""
        out: dict = {}
        for (g, k), C in self._U._attribute(torus).items():
            mvec, lam = g
            me = su2_fold(mvec[0], lam[0] - lam[1])
            kk = int(k[0])
            row = out.setdefault(me, {})
            # C is a LaurentPoly in q; attach μ^k
            for qe, z in (C._coeffs if hasattr(C, "_coeffs")
                          else C.coeffs).items():
                cur = row.get(qe)
                term = RElement(self._R, {(kk,): z})
                row[qe] = term if cur is None else cur + term
        res = {}
        for me, row in out.items():
            rl = RLaurent(self._R, {qe: r for qe, r in row.items()
                                    if not r.is_zero()})
            if rl.coeffs:
                res[me] = rl
        return Element(res)

    # ----- contract ------------------------------------------------------

    def multiply(self, a, b) -> Element:
        return self._read(self._chart(a) * self._chart(b))

    def rho(self, a) -> Element:
        return self._read(self._chart(a).rho())

    def rho_inverse(self, a) -> Element:
        return self._read(self._chart(a).rho_inverse())

    # ----- native SU(2)+N_f=1 Schur trace --------------------------------
    #
    # The rank-1 **SU(2)** measure residue (NOT the U(2) MatterURQTorus
    # measure).  The pure-U(2) Schur residue `trace_v0` carries the prefactor
    # `(q²;q²)_∞^{2N} = (q²;q²)_∞⁴` (rank 2: the SU(2) Cartan factor times the
    # decoupled U(1)-photon factor); SU(2) (rank 1) has **one fewer**
    # `(q²;q²)_∞²` factor — the ungauged central-U(1) photon (user, 2026-06-13)
    # — so the SU(2) residue is `trace_v0 / (q²;q²)_∞²`.  The central U(1) is
    # not integrated but kept as the flavour fugacity: a content×matter-factor
    # term of matter level `lk+n` surfaces at `μ^{lk+n}`, and its v₀v₁-degree
    # (central charge) is shifted to 0 before the v-residue (an odd central
    # charge is odd under the SU(2) Cartan and integrates to 0).

    def _inv_poch2(self, K: int):
        """`1/(q²;q²)_∞²` to `q^K` as `{q_exp: int}` (cached)."""
        cache = self.__dict__.setdefault("_inv_poch2_cache", {})
        r = cache.get(K)
        if r is None:
            from qpoch import qpoch_infty, _invert_series
            p = qpoch_infty(K)
            inv = _invert_series(p * p, K)
            r = {e: inv[e] for e in range(0, K + 1) if inv[e]}
            cache[K] = r
        return r

    @staticmethod
    def _central_charge(g) -> int:
        """The (homogeneous) v₀v₁-degree of a magnetic-0 residual `g`
        (`VRational`): `deg(numerator) − #poles` (each pole `(v₀−q^M v₁)`
        is central-degree 1)."""
        degs = {a + b for (a, b) in g.num._terms}
        if len(degs) != 1:
            raise RuntimeError(f"SU2Nf1Abe.trace: residual not "
                               f"central-homogeneous: degrees {sorted(degs)}")
        return degs.pop() - sum(g.den.values())

    @staticmethod
    def _shift_central(g, half: int):
        """Multiply `g` by `(v₀v₁)^{−half}` (shift every numerator monomial
        exponent down by `half` in both v₀ and v₁) — used to bring the
        central charge to 0 before the v-residue."""
        from abelianized_torus import VRational, VLaurent
        num = {(a - half, b - half): lp for (a, b), lp in g.num._terms.items()}
        return VRational(VLaurent(num, n=2), dict(g.den), n=2)

    def trace(self, a, K: int = 20):
        """Native SU(2)+N_f=1 Schur trace, μ-refined.

        Computed from this leg's own rational-quantum-torus content: the
        magnetic-0 residual `f₀` of the chart (per matter level), times the
        fundamental-hyper Schur factor `M(μ,v)`, run through the **rank-1
        SU(2)** Schur-measure residue — the pure-U(2) residue `trace_v0`
        with **one fewer** `(q²;q²)_∞²` (the ungauged photon).  The matter
        level surfaces as the flavour fugacity μ ∈ R.  Returns an
        `RPowerSeries` over `R`, certified `==` the cone/BPS SU(2)+N_f=1
        trace.

        The matter factor uses exact Habiro (`nahm_term`) coefficients,
        expanded only at the boundary; a window-stability guard escalates
        the truncation until the q≤K result is stable (so the early
        expansion never poisons the exact cancellations)."""
        from un_nf_dressed_generators import _matter_factor_levels
        from pure_un_kalgebra import trace_v0
        from laurent_poly import LaurentPoly

        f0 = self._urq_chart(a).residuals().get((0, 0))
        if not f0:
            return RPowerSeries(self._R, {}, K)
        invp2 = self._inv_poch2(K)
        Kq = K + 16

        def _div_poch2(lp: LaurentPoly) -> LaurentPoly:
            out: dict = {}
            for e, c in lp._coeffs.items():
                for e2, c2 in invp2.items():
                    if 0 <= e + e2 <= K:
                        out[e + e2] = out.get(e + e2, 0) + c * c2
            return LaurentPoly({e: c for e, c in out.items() if c})

        def evaluate(W: int) -> dict:
            Mlev = _matter_factor_levels(2, 1, W, Kq)
            out: dict = {}                     # μ_power -> {q_exp: int}
            for lk_t, f in f0.items():
                lk = int(lk_t[0])
                for n, mf in Mlev.items():
                    g = (f * mf).simplify()
                    if g.is_zero():
                        continue
                    cc = self._central_charge(g)
                    if cc % 2:
                        continue               # odd SU(2) parity → residue 0
                    g0 = self._shift_central(g, cc // 2)
                    lp = _div_poch2(trace_v0(g0, 2, K, adaptive=True))
                    if lp.is_zero():
                        continue
                    row = out.setdefault(lk + n, {})
                    for e, c in lp._coeffs.items():
                        if 0 <= e <= K and c:
                            row[e] = row.get(e, 0) + c
            return {mu: {e: c for e, c in row.items() if c}
                    for mu, row in out.items() if any(row.values())}

        prev = None
        W = 4
        while W <= 14:
            cur = evaluate(W)
            if prev is not None and prev == cur:
                coeffs: dict = {}
                for mu, row in cur.items():
                    for e, c in row.items():
                        coeffs.setdefault(e, {})[(mu,)] = c
                return RPowerSeries(self._R, {e: RElement(self._R, t)
                                              for e, t in coeffs.items()}, K)
            prev = cur
            W += 2
        raise NotImplementedError(
            f"SU2Nf1Abe.trace({a}, K={K}): matter-factor window did not "
            f"stabilise by W=14")

    def r_label_decompose(self, a):
        # labels `(m, e)` are μ-free (μ lives in R): section = label, χ₀.
        return (tuple(a), self._R.one_basis())

    def r_label_compose(self, section, r_basis_label):
        return section



if __name__ == "__main__":
    A = SU2Nf1Abe()
    print("H0·W1 =", {k: str(v) for k, v in A.multiply((1, 0), (0, 1)).terms.items()})
    print("H1·H-1 =", {k: str(v) for k, v in A.multiply((1, 1), (1, -1)).terms.items()})
    print("ρ(H0) =", {k: str(v) for k, v in A.rho((1, 0)).terms.items()})
    print("Tr(1)  =", A.trace((0, 0), 8))
    print("Tr(H0) =", A.trace((1, 0), 8))
    print("Tr(H1) =", A.trace((1, 1), 8))   # spinor sector: odd μ-powers
