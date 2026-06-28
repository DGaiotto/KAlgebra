"""su2su2_bifund_over_pure — SU(2)×SU(2) + bifundamental as an `RGKAlgebra`
wrapping pure SU(2) ⊗ pure SU(2), fully BPS-free.

The two-gauge-factor analogue of `su2_nf_over_pure.SU2NfOverPure`: build
SU(2)×SU(2) with one bifundamental on top of the (decoupled) product of two
self-contained pure-SU(2) cone K-algebras, by supplying the bifundamental
spectrum generator in closed form and integrating it out.

  * IR auxiliary = `(pure SU(2)₁ ⊗ pure SU(2)₂).add_flavour(U(1))` — the two
    decoupled pure-SU(2) cone factors (labels `(m₁,e₁,m₂,e₂)`; multiply is the
    tensor of the two BPS-free `PureSU2KAlg` multiplies, trace is the product of
    the per-factor analytic Schur traces `trace_series`) with the bifundamental's
    **baryonic U(1)** adjoined as a genuine coefficient-ring flavour `μ` (so the
    trace can keep the μ-character — this is what makes the μ-refined index
    well-defined; the previous central-"level" encoding could not, and its trace
    honest-failed on charged states).
  * S_RG = the bifundamental `(2,2)` spectrum

        Ψ = ∏_{ε₁,ε₂ ∈ {±1}} E_𝔮(μ · v₁^{ε₁} v₂^{ε₂}),

    re-expressed in SU(2)₁×SU(2)₂ characters and replaced
    `χ_{w₁}(v₁) χ_{w₂}(v₂) → F⁽¹⁾_{w₁} F⁽²⁾_{w₂}` (a 2-D Weyl peel of the scalar
    v₁,v₂ content), dressed by the baryon level `k = μ`-power:
    `[Ψ]_{((0,w₁,0,w₂), (k,))} = c_{k,(w₁,w₂)}(q)`.
  * `Γ_RG` grading = the (tame, abelian) baryon level `k` (positive cone `Z_{≥0}`,
    height 1); the flow integrates the bifundamental out, leaving pure SU(2)×SU(2)
    and a μ-refined Schur index in `R(U(1))`.

Fully BPS-free and **pure exact-FS** (no trace override): the E_𝔮 coefficients are
Habiro-exact, the auxiliary multiply routes through `PureSU2KAlg` (Wilson CG +
the `w_1`-recursion sectors), and the trace is the generic bilinear exact-FS
pairing over the `add_flavour(U(1))` auxiliary (keeping the μ-character).
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
from sunf_dilog import eq_coeff as _a             # a_n = (-1)^n q^n/(q^2;q^2)_n (= E_𝔮 coeff)


# ---------------------------------------------------------------------------
# Bifundamental matter spectrum  Ψ = ∏_{ε} E_𝔮(μ v₁^{ε₁} v₂^{ε₂}),  χχ → F⁽¹⁾F⁽²⁾.
# ---------------------------------------------------------------------------


def su2su2_bifund_matter_spectrum(cutoff: int) -> dict[tuple, HabiroElement]:
    """`S_RG = Ψ` truncated to bifundamental level `k ≤ cutoff`.

    Returns `{(0, w₁, 0, w₂, k): c_{k,(w₁,w₂)}}` — the exact Habiro coefficient
    of `F⁽¹⁾_{w₁} F⁽²⁾_{w₂}` (Wilson line of each pure SU(2)) at bifundamental
    level `k`.  `c` is the 2-D Weyl peel
    `D_k(w₁,w₂) − D_k(w₁+2,w₂) − D_k(w₁,w₂+2) + D_k(w₁+2,w₂+2)` of the scalar
    `(v₁,v₂)`-weight polynomial `D_k(W₁,W₂) = Σ ∏ a_{n_ε}` over
    `Σ_ε n_ε = k`, `W₁ = (n₊₊+n₊₋)−(n₋₊+n₋₋)`, `W₂ = (n₊₊+n₋₊)−(n₊₋+n₋₋)`.

    (The label here keeps the historical flat `(0,w₁,0,w₂,k)` form — used by the
    linear-quiver merge test and `SU2xSU2BifundOverPure._s_rg_component`, which
    relabels it onto the `add_flavour` auxiliary.)
    """
    Z = HabiroElement.zero()
    out: dict[tuple, HabiroElement] = {}
    for k in range(cutoff + 1):
        D: dict[tuple, HabiroElement] = {}
        for npp in range(k + 1):
            for npm in range(k + 1 - npp):
                for nmp in range(k + 1 - npp - npm):
                    nmm = k - npp - npm - nmp
                    c = _a(npp) * _a(npm) * _a(nmp) * _a(nmm)
                    if c.is_zero():
                        continue
                    W1 = (npp + npm) - (nmp + nmm)
                    W2 = (npp + nmp) - (npm + nmm)
                    D[(W1, W2)] = D.get((W1, W2), Z) + c
        for w1 in range(k + 1):
            for w2 in range(k + 1):
                c = (D.get((w1, w2), Z) - D.get((w1 + 2, w2), Z)
                     - D.get((w1, w2 + 2), Z) + D.get((w1 + 2, w2 + 2), Z))
                if not c.is_zero():
                    out[(0, w1, 0, w2, k)] = c
    return out


# ---------------------------------------------------------------------------
# Auxiliary: pure SU(2) ⊗ pure SU(2) (the baryon U(1) is adjoined via add_flavour).
# ---------------------------------------------------------------------------


class _PureSU2x2KAlg(KAlgebra):
    """Two decoupled pure-SU(2) cone K-algebras tensored.  Labels `(m₁,e₁,m₂,e₂)`;
    multiply tensors the two BPS-free cone multiplies (the gauge factors q-commute
    trivially), trace is the product of the per-factor analytic Schur traces
    `trace_series` (the `(m,e)` Wilson/'t Hooft trace, BPS-free).

    The bifundamental's baryonic U(1) is **not** baked in here — it is adjoined by
    `.add_flavour(AbelianZPlusRing(1))`, so the coefficient ring carries the
    μ-fugacity and the (generic) trace keeps the μ-refined index."""

    def __init__(self) -> None:
        self._c1 = PureSU2KAlg()
        self._c2 = PureSU2KAlg()

    def coefficient_ring(self):
        return self._c1.coefficient_ring()

    def identity(self):
        return (0, 0, 0, 0)

    def multiply(self, a, b):
        p1 = self._c1.multiply((a[0], a[1]), (b[0], b[1]))
        p2 = self._c2.multiply((a[2], a[3]), (b[2], b[3]))
        out: dict = {}
        for (M1, E1), c1 in p1.terms.items():
            for (M2, E2), c2 in p2.terms.items():
                out[(M1, E1, M2, E2)] = c1 * c2
        return Element(out)

    def rho(self, a):
        r1 = self._c1.rho((a[0], a[1]))
        r2 = self._c2.rho((a[2], a[3]))
        return (r1[0], r1[1], r2[0], r2[1])

    def rho_inverse(self, a):
        r1 = self._c1.rho_inverse((a[0], a[1]))
        r2 = self._c2.rho_inverse((a[2], a[3]))
        return (r1[0], r1[1], r2[0], r2[1])

    def trace(self, a, K: int = 20):
        prod = trace_series(a[0], a[1], K) * trace_series(a[2], a[3], K)
        return RPowerSeries(self.coefficient_ring(), dict(prod._coeffs), K)


# ---------------------------------------------------------------------------
# The RGKAlgebra.
# ---------------------------------------------------------------------------


class SU2xSU2BifundOverPure(RGKAlgebra):
    """SU(2)×SU(2) + one bifundamental on `(pure SU(2) ⊗ pure SU(2)).add_flavour(U(1))`,
    fully BPS-free and pure exact-FS; the two-gauge-factor analogue of
    `SU2NfOverPure`.  `S_RG = ∏_{ε} E_𝔮(μ v₁^{ε₁} v₂^{ε₂})` peeled to χ_{w₁}χ_{w₂}
    (Wilson of each pure SU(2)); the baryon level `k` is the `Γ_RG` grading and the
    `add_flavour` U(1), so the trace is the μ-refined Schur index."""

    def __init__(self) -> None:
        self._aux = _PureSU2x2KAlg().add_flavour(AbelianZPlusRing(1))

    def auxiliary(self):
        return self._aux

    def grading(self):
        """`Γ_RG = Z` = the baryon level `k` (the `add_flavour` U(1) charge);
        positive cone `Z_{≥0}`, height 1."""
        return Grading(rank=1, deg=lambda lab: (lab[1][0],), height=(1,),
                       cone_gens=((1,),))

    def _s_rg_component(self, p):
        """`[Ψ]_p` — exact graded component at baryon level `p=(k,)`, relabelled
        onto the `add_flavour` auxiliary `((0,w₁,0,w₂), (k,))`; `{}` off the cone."""
        k = int(p[0])
        if k < 0:
            return {}
        return {((m1, w1, m2, w2), (k,)): c
                for (m1, w1, m2, w2, kk), c in su2su2_bifund_matter_spectrum(k).items()
                if kk == k}

    def rg_generator(self, cutoff: int) -> dict[tuple, HabiroElement]:
        """`Ψ` windowed to baryon level `k ≤ cutoff`."""
        out: dict = {}
        for k in range(cutoff + 1):
            out.update(self._s_rg_component((k,)))
        return out

    def _section_split(self, label):
        """Auxiliary labels are `((m₁,e₁,m₂,e₂), (k,))` — the gauge part is the
        section, the baryon `k` the (central, additive) flavour; the SU(2) Wilson
        content fuses by Clebsch–Gordan inside the section, so disable the
        flavour-shift multiply cache (`flav = None`) and let the generic
        `from_ir_image(RG(a)·RG(b))` route through `PureSU2KAlg`."""
        return tuple(label), None

    def __repr__(self) -> str:
        return "SU2xSU2BifundOverPure(SU(2)×SU(2)+bifund over pure SU(2)⊗pure SU(2))"


if __name__ == "__main__":
    import warnings
    A = SU2xSU2BifundOverPure()
    print(repr(A))
    print("  exact-FS available:", A._fs_exact_available())
    print("  S_RG (baryon level ≤ 1):")
    for lab in sorted(A.rg_generator(1)):
        print(f"    {lab}:  {A.rg_generator(1)[lab]}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = A.trace(A.identity(), 4)
        print("  μ-refined vacuum index:",
              {e: str(r) for e, r in sorted(v.coeffs.items()) if str(r) not in ("0", "")},
              " warns =", len(w))
