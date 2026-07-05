"""`SU2UNf{2,3,4}AbeKAlgebra` — `A_𝖖[SU(2)+N_f]` with **U(N_f) flavour**,
built by the principled **ungauging** route (user direction, 2026-06-13).

Take the U(2)+N_f abelianized algebra `UNNfKAlgebra(2, N_f)` — which has a
certified iso to BPS — and **ungauge the diagonal U(1)** of the gauge
U(2).  This is the correct construction for `SU(2)` with general `N_f` and
**U(N_f) flavour**; promoting `U(N_f)` to the full `SO(2N_f)` flavour is
the separately-harder step (the spinor Z₂ of `U(2)=(SU(2)×U(1))/Z₂`; see
"Status").

Construction
------------
`SU2UNf{K}AbeKAlgebra = UngaugedKAlgebra(UNNfKAlgebra(2,K), E, …)` where
`E = (((0,0),(1,1)), ())` is the **determinant Wilson** — the electric
generator of the diagonal U(1) ⊂ U(2).  The ungauged algebra is the
**centralizer of `E`** = the magnetically-`E`-neutral sector = **traceless
magnetic cocharacters `(m,−m)`** = the SU(2) gauge sector.  `E` is promoted
to a U(1) flavour fugacity `z` (the baryonic/topological U(1)), and
ungauging **restores the U(1) vector-multiplet (photon) measure**
`(𝖖²;𝖖²)_∞²` in the trace.  Coefficient ring:

    R  =  R(SU(N_f)) ⊗ R(U(1))  =  U(N_f),

the matter `SU(N_f)` (from `UNNfKAlgebra`) tensored with the ungauged U(1).

Trace
-----
`UngaugedKAlgebra.trace` keeps only the gauged flavour-singlet; the
override here keeps the full `SU(N_f)` content (tensored with `z^n`):

    Tr_ung(a)(SU(N_f), z) = [ Σ_n z^n · Tr_{U(2)+N_f}(a·E^n) ] / (𝖖²;𝖖²)_∞².

Validation
----------
For `N_f=2`, identified with BPS (`build_bps_su2_nf2`, Spin(4) Cartan
`(μ_L,μ_R)`) via `SU(2)_F ↔ μ_L`, `z ↔ μ_R²`, the ungauged `Tr(1)`
**matches the BPS Spin(4) index exactly** (the full SO(4) adjoint
`(3,1)+(1,3)` recovered from the U(2) data).  `N_f=3` is checked against
`build_bps_su2_nf3` analogously; `N_f=4` has no simple BPS spectrum
generator (it is validated structurally — ring/centralizer/closure).

`U(N_f) → SO(2N_f)` flavour enhancement (`so2nf_index`)
-------------------------------------------------------
The matter flavour of `SU(2)+N_f` is the full `SO(2N_f)` (the pseudoreal
fundamental ⇒ the `2N_f` half-hypers form the `SO(2N_f)` vector), with
`U(N_f) ⊂ SO(2N_f)` the **maximal-rank** subgroup (`SO` vector =
`N_f ⊕ N̄_f`).  `so2nf_index` recovers it as a **recognize-after wrapper**
(the answer to "wrapper or URQTorus surgery?": a wrapper — the `U(N_f)`
presentation is already correct, `SO(2N_f)` is purely a trace-level flavour
reassembly).

The reassembly reads the **raw `U(1)^{N_f}` flavour-Cartan weights `k⃗`**
straight off each chart trace, summed over the baryonic det-power tower
(`Σ_n Tr(a·det^n)`); gauge invariance locks `Σ k = −2n`, so the tower sum is
the baryon-graded index, and the `k⃗` *are* the `SO(2N_f)` orthogonal weights
up to the **spinor shift** `s = m/2` (`x_i = k_i − s`; `m` the `SU(2)`
magnetic charge — `s=0` tensor sector, `s=½` bare monopole).  Apply the
photon measure, un-branch via `so2nf_characters`.  Crucially **`𝖖` stays
integer** — the only half-integers are the `SO`-spinor *flavour* weights
(`det^{1/2}`), per the v₀v₁-commuting restriction (centralizer of `E=det`):
no `𝖖^{1/2}` double-cover.  *(Subtlety that makes this work for all `N_f`:
the `SU(N_f)`-peeled trace specializes the flavour central `U(1)` to `1`
(ruling D8b), discarding the baryon that separates the simple-`SO(2N_f)`
reps; the raw `k⃗` keep it.  For `N_f=2` the `SO(4)=SU(2)×SU(2)` product let
the gauge det-power substitute, which is why the SU-peel route sufficed
there — but not for `N_f≥3`.)*

Status / scope
--------------
* **N_f=2 (SO(4)): validated against BPS.**  `so2nf_index(1)` → genuine
  tensor reps (`q⁰` vacuum, `q²` adjoint `(1,±1)=(3,1)+(1,3)` currents, …),
  reconstructing `build_bps_su2_nf2`'s index exactly through `q⁴`;
  `so2nf_index(monopole)` → the SO(4) **spinor** `(½,½)` (half-integer
  flavour weight, virtual/signed), matching the centred BPS `γ₃` trace.
* **N_f=3,4 (SO(6), SO(8)): working.**  `so2nf_index(1)` → vacuum then the
  `SO(2N_f)` adjoint `(1,1,0,…)` currents (genuine); `so2nf_index(monopole)`
  → the `SO(2N_f)` **spinor** `(½,…,±½)` (half-integer flavour weights, `𝖖`
  integer).  `W(D_{N_f})`-invariance is checked at every order.
* `UNNfKAlgebra` gauged trace: complete to `K=4` for the det-powers used at
  `N_f≤4` (raising `K` can *worsen* the SU-peel; the raw-`k⃗` route here only
  uses the chart trace, so it is unaffected).  `so2nf_index` raises if an
  order fails validity / `W(D_{N_f})`-invariance (incomplete trace).
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

from zplus_ring import RElement, RPowerSeries
from ungauge_kalgebra import UngaugedKAlgebra


__all__ = [
    "SU2UNfAbeKAlgebra",
    "SU2UNf2AbeKAlgebra",
    "SU2UNf3AbeKAlgebra",
    "SU2UNf4AbeKAlgebra",
]


# The determinant Wilson of U(2): the diagonal-U(1) electric generator
# (flavour-trivial `w = ()`, the same for every N_f ≥ 2).
_DET_E = (((0, 0), (1, 1)), ())


def _det_charge(label) -> int:
    """Diagonal-U(1) electric charge of a `UNNfKAlgebra(2,N_f)` label
    `((m, λ), w)` — the det charge `λ₁ + λ₂`."""
    return label[0][1][0] + label[0][1][1]


def _det_shift(label, n: int):
    """`label · E^n` = raise the Levi rep by `(n, n)` (det^n)."""
    (m, lam), w = label
    return ((m, (lam[0] + n, lam[1] + n)), w)


def _int_or_half(x):
    """Normalize an exact rational SO-weight coordinate: an integral value
    as `int`, otherwise a `Fraction` (the half-integer spinor weights)."""
    from fractions import Fraction
    fr = Fraction(x)
    return int(fr) if fr.denominator == 1 else fr


def _valid_so_weight(x) -> bool:
    """Whether `x` is a valid `SO(2N_f)` weight in the orthogonal `e_i`
    basis: **all-integer** (tensor) or **all-half-integer** (spinor)."""
    from fractions import Fraction
    dens = {Fraction(xi).denominator for xi in x}
    return dens <= {1} or dens == {2}


class SU2UNfAbeKAlgebra(UngaugedKAlgebra):
    """`A_𝖖[SU(2)+N_f]` with U(N_f) flavour — ungauge the diagonal U(1) of
    `UNNfKAlgebra(2, N_f)`.  See the module docstring."""

    def __init__(self, Nf: int) -> None:
        from un_nf_kalgebra import UNNfKAlgebra
        if Nf < 2:
            raise ValueError("SU2UNfAbeKAlgebra: N_f >= 2 (N_f=1 is SU2Nf1Abe)")
        self._Nf = int(Nf)
        G = UNNfKAlgebra(2, self._Nf)
        super().__init__(G, _DET_E, _det_charge,
                         e_shift=_det_shift, measure_power=2)

    @property
    def Nf(self) -> int:
        return self._Nf

    # ----- the traceless SU(2) sector (centralizer reps) ----------------
    def monopole(self, m: int, lam=(0, 0), w=()):
        """A traceless-magnetic SU(2) line `((m, −m), λ; w)` (in the
        centralizer of the det Wilson).  `m=0` ⇒ Wilson/matter sector."""
        return ((m, -m), tuple(lam)), tuple(w)

    # ----- trace: keep the full SU(N_f) flavour (override) --------------
    def trace(self, a, K: int = 4) -> RPowerSeries:
        """U(N_f) flavoured trace `Σ_n z^n·Tr_{U(2)+N_f}(a·E^n)` divided by
        the restored photon measure, **keeping the gauged `SU(N_f)`
        content** (`UngaugedKAlgebra.trace` would drop it to the singlet).
        Returns an `RPowerSeries` over `R(SU(N_f)) ⊗ R(U(1))`."""
        R = self._R
        acc: dict = {}                       # fqexp -> {(su_w, z): coeff}
        for n in range(-(K + 1), K + 2):
            tg = self._G.trace(self._e_shift(a, n), K)
            terms = tg._coeffs if hasattr(tg, "_coeffs") else tg.coeffs
            for e, rc in terms.items():
                rterms = rc.terms if hasattr(rc, "terms") else {(): int(rc)}
                for sw, c in rterms.items():
                    if c:
                        acc.setdefault(e, {})
                        acc[e][(sw, n)] = acc[e].get((sw, n), 0) + int(c)
        inv = self._inv_measure(K)
        out: dict = {}
        for e, zd in acc.items():
            for fe, fc in inv.items():
                if e + fe > K:
                    continue
                for (sw, zp), c in zd.items():
                    v = c * fc
                    if not v:
                        continue
                    key = (sw, (zp,))
                    cur = out.get(e + fe)
                    d = dict(cur.terms) if cur is not None else {}
                    d[key] = d.get(key, 0) + v
                    out[e + fe] = RElement(R, d)
        out = {e: r for e, r in out.items()
               if any(v for v in r.terms.values())}
        return RPowerSeries(R, out, K)

    # ----- SO(2 N_f) flavour enhancement (recognize-after wrapper) ------
    def so_flavour_ring(self):
        """The **enhanced** flavour ring `R(Spin(2 N_f))` (`SO2NfZPlusRing`)
        — the full matter flavour of `SU(2)+N_f`, with `U(N_f) ⊂ SO(2 N_f)`
        the maximal-rank subgroup.  Recognized on the index by
        `so2nf_index`."""
        from so2nf_characters import SO2NfZPlusRing
        return SO2NfZPlusRing(self._Nf)

    def _spinor_shift(self, a):
        """The per-coordinate `SO(2 N_f)` Cartan shift `s = m/2` (`m` the
        `SU(2)` magnetic charge of `a`): `s = 0` on the **tensor** (vacuum /
        Wilson) sector, `s = ½` on the bare **monopole** — the SO-spinor
        weights are `det^{1/2}`-shifted **flavour** fugacities, so `𝖖` stays
        integer (the half-integers are flavour-only, per the v₀v₁-commuting
        restriction; the `Z₂` of `U(2)=(SU(2)×U(1))/Z₂` = the tensor/spinor
        split)."""
        from fractions import Fraction
        return Fraction(a[0][0][0], 2)

    def so2nf_index(self, a, K: int = 4) -> dict:
        """**`SO(2 N_f)`-enhanced index** of `a` (recognize-after wrapper).

        The full matter flavour of `SU(2)+N_f` is `SO(2 N_f)`, with
        `U(N_f) ⊂ SO(2 N_f)` maximal-rank (`SO` vector `= N_f ⊕ N̄_f`).  This
        is the answer to "wrapper or URQTorus surgery?" — a **wrapper**: the
        `U(N_f)` presentation is already correct; `SO(2 N_f)` is purely a
        trace-level flavour reassembly.

        Method (the raw-Cartan route — works for all `N_f`).  Sum the gauged
        trace over the baryonic det-power tower, reading the **raw
        `U(1)^{N_f}` flavour-Cartan weights `k⃗`** directly off each chart
        trace (`self._G.chart(a·det^n).trace`) — *not* the `SU(N_f)`-peeled
        content, which specializes the flavour central `U(1)` to `1` (ruling
        D8b) and so discards exactly the baryon that distinguishes the
        `SO(2 N_f)` reps.  Gauge invariance locks the flavour baryon to the
        det charge, `Σ k = −2n`, so the tower sum *is* the baryon-graded
        index.  The `k⃗` are the `SO(2 N_f)` orthogonal `e_i`-basis weights up
        to the **spinor shift** `s = m/2` (`x_i = k_i − s`); apply the photon
        measure `1/(𝖖²;𝖖²)_∞^{p}`, then un-branch via
        `so2nf_characters.decompose`.  **`𝖖` stays integer** — the only
        half-integers are the spinor flavour weights `x_i`.

        Returns `{fqexp: {dominant Spin(2 N_f) weight λ: multiplicity}}` —
        **genuine `≥ 0`** for the vacuum index `a = 1` (vacuum, then the
        `SO(2 N_f)` adjoint `(1,1,0,…)` currents, …), **virtual** (signed) for
        a graded insertion (a monopole lands in the `SO(2 N_f)` **spinor**).

        Validated against BPS at `N_f=2` (`Tr(1)` reconstructs
        `build_bps_su2_nf2` exactly; monopole → SO(4) spinor) and against the
        flavour-character theory at `N_f=3,4` (`Tr(1)` → `SO(2N_f)` adjoint;
        monopole → the spinor).  Raises `ValueError` if a regraded order is
        not integral/half-integral or not `W(D_{N_f})`-invariant — an honest
        failure should the gauged trace be incomplete at the chosen `K`."""
        from so2nf_characters import is_weyl_invariant, decompose

        Nf = self._Nf
        s = self._spinor_shift(a)

        # Σ_n Tr(a·det^n): raw U(1)^{N_f} flavour-Cartan weights k⃗, shifted
        # to the SO Cartan x_i = k_i − s  (pre-measure).
        orth: dict = {}
        for n in range(-(K + 1), K + 2):
            levels = self._G.chart(self._e_shift(a, n)).trace(K=K)
            for kv, C in levels.items():
                cf = C._coeffs if hasattr(C, "_coeffs") else getattr(C, "coeffs", {})
                x = tuple(_int_or_half(ki - s) for ki in kv)
                for e, coeff in cf.items():
                    if 0 <= e <= K and coeff:
                        d = orth.setdefault(e, {})
                        d[x] = d.get(x, 0) + int(coeff)

        # restore the photon (vector-multiplet) measure
        inv = self._inv_measure(K)
        meas: dict = {}
        for e, d in orth.items():
            for fe, fc in inv.items():
                if e + fe > K:
                    continue
                for x, c in d.items():
                    v = c * fc
                    if v:
                        dd = meas.setdefault(e + fe, {})
                        dd[x] = dd.get(x, 0) + v

        # recognize Spin(2 N_f) content per fq order
        out: dict = {}
        for e in sorted(meas):
            poly = {x: v for x, v in meas[e].items() if v}
            if not poly:
                continue
            if not all(_valid_so_weight(x) for x in poly):
                raise ValueError(
                    f"so2nf_index(N_f={Nf}): q^{e} weights are not "
                    f"integral/half-integral SO({2 * Nf}) weights — gauged "
                    f"trace likely incomplete at K={K}.")
            if not is_weyl_invariant(Nf, poly):
                raise ValueError(
                    f"so2nf_index(N_f={Nf}): the q^{e} index is not "
                    f"W(D_{Nf})-invariant — SO({2 * Nf}) enhancement not "
                    f"recognized (gauged trace likely incomplete at K={K}).")
            out[e] = decompose(Nf, poly)
        return out

    def __repr__(self) -> str:
        return f"SU2UNf{self._Nf}AbeKAlgebra (ungauged U(2)+N_f={self._Nf})"


class SU2UNf2AbeKAlgebra(SU2UNfAbeKAlgebra):
    """`A_𝖖[SU(2)+N_f=2]`, U(2) flavour (validated `Tr(1)==BPS`)."""
    def __init__(self) -> None:
        super().__init__(2)


class SU2UNf3AbeKAlgebra(SU2UNfAbeKAlgebra):
    """`A_𝖖[SU(2)+N_f=3]`, U(3) flavour."""
    def __init__(self) -> None:
        super().__init__(3)


class SU2UNf4AbeKAlgebra(SU2UNfAbeKAlgebra):
    """`A_𝖖[SU(2)+N_f=4]`, U(4) flavour (no simple BPS S; structural)."""
    def __init__(self) -> None:
        super().__init__(4)


if __name__ == "__main__":
    for cls in (SU2UNf2AbeKAlgebra, SU2UNf3AbeKAlgebra, SU2UNf4AbeKAlgebra):
        A = cls()
        print(f"{cls.__name__}: flavour {A.coefficient_ring()}")
        print(f"   centralizer(1)={A.in_centralizer(A.identity())}, "
              f"centralizer(H0)={A.in_centralizer(A.monopole(1))}")
        print(f"   Tr(1) = {A.trace(A.identity(), 4)}")
