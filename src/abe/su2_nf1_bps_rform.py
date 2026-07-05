"""`SU2Nf1BpsRForm` — SU(2)+N_f=1 as the **R-form of the BPS realization**:
the BPS quantum-torus chart re-presented with the flavour in the
coefficient ring (2026-06-13).

**What this is — and what it is NOT.**  This is *not* an `AbeKAlgebra`
and does *not* use a `URQTorus` / enriched rational quantum torus.  It is
a thin façade over the **BPS realization** `build_bps_su2_nf1()` (a
`BPSKAlgebra`, the BPS-quiver quantum-torus chart on the **integer**
charge lattice): `multiply` / `rho` / `rho_inverse` / `trace` all
delegate to that `BPSKAlgebra`.  The only thing it adds is folding the
BPS `ker(B)` flavour label (the 3rd charge coordinate `f`) into the
coefficient ring `R = AbelianZPlusRing(1)` — i.e. it presents the same
algebra as `bps` but with the flavour as `μ ∈ R` instead of in the
label.

So relative to the object's `bps` leg it differs *only* in flavour
presentation (R-form vs Z-form); it is a correct-flavour SU(2)+N_f=1
KAlgebra (certified against cone/bps), but it is **not** the genuine
abelianized / rational-quantum-torus presentation.

The genuine `AbeKAlgebra` realization — on the `URQTorus` substrate,
supplying `torus_shape` / `chart` / `decompose`, the actual rational
quantum torus — is the recorded build: ungauge a from-scratch
`[SU(2)×U(1)]+hyper` base (RGKAlgebra over `pure SU(2) ⊗ pure U(1)` with
matter `S_RG = E_q(vw)E_q(w/v)`, then `UngaugedKAlgebra` at the U(1)
Wilson `w`).  Same algebra; it would actually live on a rational quantum
torus.

Correct flavour (inherited from the BPS chart): every `H_n`, `W_n` is at
μ⁰ and μ appears only in the monopole×Wilson → meson OPE
(`H0·W1 = q·H1 + q⁻¹·H-1 + μ·1`); ρ μ-shifts; generators carry no
flavour charge (the flavoured-KAlgebra contract).
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
from zplus_ring import AbelianZPlusRing, RElement, RLaurent


__all__ = ["SU2Nf1BpsRForm"]


class SU2Nf1BpsRForm(KAlgebra):
    """SU(2)+N_f=1 = the BPS chart with the `ker(B)` flavour folded to R.

    A façade over `BPSKAlgebra` (`build_bps_su2_nf1`); NOT an
    `AbeKAlgebra` and NOT on a `URQTorus`.  See the module docstring."""

    def __init__(self) -> None:
        from bps_su2_nf1 import build_bps_su2_nf1
        self._bps = build_bps_su2_nf1()
        self._R = AbelianZPlusRing(rank=1)

    # ----- KAlgebra basics -----------------------------------------------

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return (0, 0)

    # ----- the f → μ fold -------------------------------------------------

    def _lift(self, a):
        """label `(x, y)` → BPS charge `(x, y, 0)` (f = 0)."""
        return (a[0], a[1], 0)

    def _fold_element(self, P: Element) -> Element:
        """BPS `Element` `{(x,y,f): LaurentPoly}` → `{(x,y): RLaurent}`
        with the `ker(B)` charge `f` folded onto the fugacity `μ^f ∈ R`."""
        acc: dict = {}
        for lab, lp in P.terms.items():
            x, y, f = lab
            coeffs = lp._coeffs if hasattr(lp, "_coeffs") else lp.coeffs
            d = acc.setdefault((x, y), {})
            for qe, c in coeffs.items():
                cur = d.get(qe)
                term = RElement(self._R, {(f,): c})
                d[qe] = term if cur is None else cur + term
        out = {}
        for key, d in acc.items():
            rl = RLaurent(self._R, {qe: r for qe, r in d.items()
                                    if not r.is_zero()})
            if rl.coeffs:
                out[key] = rl
        return Element(out)

    # ----- contract ------------------------------------------------------

    def multiply(self, a, b) -> Element:
        return self._fold_element(
            self._bps.multiply(self._lift(a), self._lift(b)))

    def rho(self, a) -> Element:
        x, y, f = self._bps.rho(self._lift(a))
        return Element({(x, y): RLaurent(self._R,
                                         {0: RElement(self._R, {(f,): 1})})})

    def rho_inverse(self, a) -> Element:
        x, y, f = self._bps.rho_inverse(self._lift(a))
        return Element({(x, y): RLaurent(self._R,
                                         {0: RElement(self._R, {(f,): 1})})})

    def trace(self, a, K: int = 20):
        """The BPS μ-refined Schur index of the line (already over R)."""
        return self._bps.trace(self._lift(a), K)

    def r_label_decompose(self, a):
        # labels `(x, y)` are μ-free (μ lives in the coefficients): section
        # = label, χ₀.
        return (tuple(a), self._R.one_basis())

    def r_label_compose(self, section, r_basis_label):
        return section



if __name__ == "__main__":
    A = SU2Nf1BpsRForm()
    print("H0·W1 =", {k: str(v) for k, v in A.multiply((1, 0), (0, -1)).terms.items()})
    print("H1·H-1 =", {k: str(v) for k, v in A.multiply((0, 1), (1, -1)).terms.items()})
    print("ρ(H0) =", {k: str(v) for k, v in A.rho((1, 0)).terms.items()})
