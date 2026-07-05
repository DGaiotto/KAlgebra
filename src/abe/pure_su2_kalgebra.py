"""`PureSU2KAlgebra` — pure SU(2) as an `AbeKAlgebra` on the group-general
**WRQTorus** substrate (`root_datum.su_2()`, ω-basis, d = 1).

The pure-U(N) keystone leans on a luxury SU(N) lacks: every fundamental coweight
is **minuscule**, so dressed minuscule monopoles are closed-form bar-invariant
generators and every canonical is a polynomial in them.  Restricting pure U(2) to
the **trace-zero magnetic sublattice** `m·(1,−1)` = SU(2) (the central U(1)/det
frozen) loses the minuscules: the simplest monopole is the **adjoint** `H = L_{1,0}`,
which **bubbles**.

The build is **fully native on `WRQTorus(su_2())`** (D9/D10: WRQTorus universally).
Labels `(m, e)` in the `su2_fold` frame, unflavoured (`R = Z`).  The single irreducible input is the
**adjoint-monopole fiber `{L_{1,e}}`** (elementary closed form `_seed_dyon1`, in
ω-coordinates `v² = v₀/v₁`):

    atom (1):  v^e ,   atom (−1):  v^{−e} ,
    atom (0):  (q+q⁻¹)·v²/((1−q²v²)(1−q⁻²v²))          (e even)
               (v³+v) /((1−q²v²)(1−q⁻²v²))              (e odd);

everything else:

    L_{0,e}        =  wrq_torus.wilson  (residual-native χ_e — Clebsch–Gordan free)
    L_{m,e} (m≥2)  =  the generic wrq_torus.peel_to_canonical on the well-founded
                      seed `H·L_{m−1,e}` + kl_bar_correct  (the datum-general
                      product-and-peel; the engine lives in `wrq_torus`, this class
                      supplies only the seed + the label frame)
    L_{m,e<0}      =  ι(L_{m,-e}),  ι = v↦1/v          (electric charge conjugation)

Never an in-presentation solve (the constructive-build rule); honest-fails on a
cyclic dependency or non-terminating peel.  Certified `==` the BPS Kronecker chart
(`pure_su2_bps_iso`) on multiply / ρ / trace / orthonormality (Goal 2.1), and
`well_formed` (W1+W2 KL acceptance) on every chart.

This is the SU(2) instance of Plan 24's "U(N)→SU(N) microscope": the single input
is the adjoint-monopole bubbling, everything else native.  The general-`N` engine
(`PureSUNKAlgebra`) takes the same shape with the adjoint fiber from the
`PureUNKAlgebra(N)` oracle.

Conventions: 't Hooft–Wilson labels `(m, e)` (`m ≥ 0`; `e ≥ 0` when `m = 0`), `det`
collapsed (pure SU(2) is unflavoured, `TrivialZPlusRing`).
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

from abe_kalgebra import AbeKAlgebra
from kalgebra import Element, Label
from laurent_poly import LaurentPoly
from zplus_ring import ZPlusRing, TrivialZPlusRing, RPowerSeries
from root_datum import su_2
from weyl_torus_ring import TorusLaurent, TorusRational
from wrq_torus import (WRQTorus, wilson, decompose as _wrq_decompose,
                       peel_to_canonical, kl_bar_correct)


__all__ = ["PureSU2KAlgebra", "su2_fold"]


def su2_fold(m: int, e: int) -> tuple[int, int]:
    """Fold `(m, e)` into the SU(2) Weyl chamber: Weyl acts by `(m,e) ↦ (-m,-e)`,
    pick `m ≥ 0`, and at `m = 0` pick `e ≥ 0`."""
    if m < 0:
        m, e = -m, -e
    if m == 0 and e < 0:
        e = -e
    return (m, e)


class PureSU2KAlgebra(AbeKAlgebra):
    """Pure SU(2) on `WRQTorus(su_2())` — labels `(m, e)` in the `su2_fold`
    frame, unflavoured (`R = Z`).  Native build: the adjoint-monopole seed
    `_seed_dyon1` + the generic `wrq_torus` product-and-peel."""

    _MAX_PEEL = 400

    def __init__(self) -> None:
        self.datum = su_2()
        self._R = TrivialZPlusRing()
        self._cache: dict[tuple[int, int], WRQTorus] = {}
        self._building: set[tuple[int, int]] = set()

    # ----- contract: coefficient ring / identity / shape / sections ----------
    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self) -> Label:
        return (0, 0)

    def torus_shape(self):
        """One native SU(2) node, no fundamentals — the honest root-datum
        shape (D6 re-ruling 2026-07-02)."""
        from abe_kalgebra import TorusShape
        return TorusShape.from_root_data((self.datum,))

    def _label_section_decompose(self, label: Label):
        return (label, self.coefficient_ring().one())

    # ----- the m=1 seed (the single irreducible input) -----------------------
    def _seed_dyon1(self, e: int) -> WRQTorus:
        """`L_{1,e}` (e ≥ 0) in ω-coordinates — the adjoint-monopole fiber
        closed form (module docstring); bar-invariant by construction."""
        d = self.datum
        den = {((2,), 2): 1, ((2,), -2): 1}
        if e % 2 == 0:
            num = TorusLaurent(d, {(2,): LaurentPoly({-1: 1, 1: 1})})
        else:
            num = TorusLaurent(d, {(3,): LaurentPoly({0: 1}),
                                   (1,): LaurentPoly({0: 1})})
        f0 = TorusRational(d, num, den)
        return WRQTorus(d, {
            (1,): TorusRational.from_laurent(
                TorusLaurent(d, {(e,): LaurentPoly({0: 1})})),
            (-1,): TorusRational.from_laurent(
                TorusLaurent(d, {(-e,): LaurentPoly({0: 1})})),
            (0,): f0,
        })

    # ----- ι: electric charge conjugation (v ↦ 1/v, atoms fixed) -------------
    @staticmethod
    def _v_invert(x: WRQTorus) -> WRQTorus:
        return WRQTorus(x.datum, {m: f.vinv() for m, f in x.residuals().items()})

    # ----- the build ----------------------------------------------------------
    def _lower_chart(self, mt, et) -> WRQTorus:
        """The already-built canonical for a recognized (datum-frame) label —
        the `lower_chart` callback of the generic builder."""
        return self._build(*su2_fold(mt[0], et[0]))

    def _build(self, m: int, e: int) -> WRQTorus:
        key = (m, e)
        if key in self._cache:
            return self._cache[key]
        if key in self._building:
            raise NotImplementedError(
                f"PureSU2KAlgebra._build: cyclic dependency at L_{key}")
        self._building.add(key)
        try:
            if e < 0:
                result = self._v_invert(self._build(m, -e))
            elif m == 0:
                result = wilson(self.datum, (e,))
            elif m == 1:
                result = self._seed_dyon1(e)
            else:
                seed = self._build(1, 0) * self._build(m - 1, e)
                L = peel_to_canonical(self.datum, ((m,), (e,)), seed,
                                      self._lower_chart, self._MAX_PEEL)
                result = kl_bar_correct(self.datum, L, ((m,), (e,)),
                                        self._lower_chart, self._MAX_PEEL)
        finally:
            self._building.discard(key)
        self._cache[key] = result
        return result

    # ----- contract triple ----------------------------------------------------
    def chart(self, label: Label):
        m, e = su2_fold(*label)
        return self._build(m, e)

    def decompose(self, x) -> Element:
        out = _wrq_decompose(
            self.datum, x,
            build=lambda dat, mv, ev: self._build(*su2_fold(mv[0], ev[0])))
        folded: dict = {}
        for (mv, ev), C in out.items():
            lab = su2_fold(mv[0], ev[0])
            folded[lab] = folded.get(lab, LaurentPoly.zero()) + C
        return Element({lab: C for lab, C in folded.items() if not C.is_zero()})

    # ----- ρ: read the image label via well_formed ---------------------------
    def _wf_label(self, img, what: str) -> Label:
        wf = img.well_formed()
        if wf is False:
            raise NotImplementedError(f"PureSU2KAlgebra.{what}: image not "
                                      f"well-formed")
        mv, ev = wf
        return su2_fold(mv[0], ev[0])

    def rho(self, a: Label) -> Label:
        return self._wf_label(self.chart(a).rho(), "rho")

    def rho_inverse(self, a: Label) -> Label:
        return self._wf_label(self.chart(a).rho_inverse(), "rho_inverse")

    # ----- trace / inner: WRQ LaurentPoly (pure gauge, unflavoured) ----------
    def trace(self, a: Label, K: int = 20) -> RPowerSeries:
        """`Tr(L_a)` — the native pure-SU(2) Schur trace of the chart (validated
        `==` the BPS Kronecker leg)."""
        lp = self.chart(a).trace(K)
        return RPowerSeries(self._R,
                            {e: c for e, c in lp._coeffs.items() if 0 <= e <= K},
                            K)

    def inner_product(self, a: Label, b: Label, K: int = 20) -> RPowerSeries:
        """`I_{a,b} = Tr(ρ(L_a)·L_b)` — the §6b chart pairing with the SU(2)
        measure."""
        lp = self.chart(a).inner(self.chart(b), K)
        return RPowerSeries(self._R,
                            {e: c for e, c in lp._coeffs.items() if 0 <= e <= K},
                            K)


if __name__ == "__main__":
    A = PureSU2KAlgebra()
    print("pure SU(2) on WRQTorus(su_2()); labels (m, e)")
    print("  H·H =", {k: str(v) for k, v in A.multiply((1, 0), (1, 0)).terms.items()})
    print("  ρ(H) =", A.rho((1, 0)))
    print("  Tr(1, 6) =", dict(A.trace((0, 0), 6).coeffs))
