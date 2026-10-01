"""`U2Nf1QTConeKAlg` — U(2)+N_f=1 (no flavour) as a **`QTCone`
`ConeKAlgebra`**: the matter sibling of
`pure_u2_qtcone.PureU2QTConeKAlg`.

The physics this realisation exhibits — **the hyper screens the central
monopole, reducing the central torus from rank 2 to rank 1**:

* In *pure* U(2) the central `U(1)` photon is free, giving a rank-2
  quantum-torus direction `{det^{±1}, w2^{±1}}` (both invertible).
* Adding one fundamental hyper makes the central `U(1)` *dynamical*: the
  hyper dresses the central 't Hooft monopole, so `det·det⁻¹` is no longer
  `1` (it is `1 + q·χ + q²·χ_det`, the meson tower) — **`det` is demoted
  from a torus generator to a conventional (matter-dressed) generator**.
  The central det-**Wilson** `w2` stays group-like (`w2·w2⁻¹ = 1`), so the
  central torus survives at **rank 1**: `torus_gens = {w2^{±1}}`.

No flavour
----------
At N_f=1 the genuine flavour group is `SU(N_f)=SU(1)=trivial` (D5/D8b — the
hyper's U(1)_F is inside the U(2) gauge centre), so this is the native
trivial-flavour algebra — `roster('u2-nf1').forget()` since 2026-09-19, the
retired `UNNfKAlgebra(2,1)` before (coefficient ring `Z`).  No `μ`
fugacity anywhere.

Cone structure (certified against the keystone `UNNfKAlgebra(2,1)`, retired
2026-09-19; the general tier's U(2)+1 is its replacement):

* **conventional gens** — `E = L_{((1,0),(0,0))}`, `F = L_{((0,-1),(0,0))}`
  (minuscule monopoles), `det = L_{((1,1),(0,0))}` and `det⁻¹` (now
  matter-dressed, not invertible), and the SU(2) Wilson
  `W = L_{((0,0),(1,0))}` (fusing);
* **torus gens** — `w2 = L_{((0,0),(1,1))}` (central det-Wilson) and its
  inverse `w2⁻¹` — the surviving rank-1 central photon torus.

Generator cocycles `L_g L_h = q^{c} L_{normal}` (certified):
`cocycle(w2, E) = −1`, `cocycle(w2, det) = −2`, `cocycle(w2, W) = 0`,
`cocycle(det, E) = 0`.  The meson seam `E·F = L_{(1,−1)}` is Pauli-blocked
to a single term at N_f=1.

Realisation policy
------------------
`multiply` is **self-computing** on the q-commuting QTCone sector — the
positive-magnetic monomials `E^{m₁−m₂}·det^{m₂}` (`m₁ ≥ m₂ ≥ 0`) dressed by
the rank-1 central `w2` torus (`λ₁ = λ₂`) — via `cone_data().derived_multiply`
(cocycles + the `w2·w2⁻¹ = 1` collapse), no keystone call; that sector is
closed and the cone-derived products match the keystone (certified,
the suite in the source repository).  Two directions fall outside it and are
taken from the U(2)+1 abelianized algebra (`U2Nf1FlavourFree`, the general
tier's `roster('u2-nf1')` with flavour forgotten; the retired `UNNfKAlgebra(2,1)`
engine until 2026-09-19): the **matter-screened
`det⁻¹`** (`det·det⁻¹` is the meson tower, not `1`) and the fusing SU(2)
Wilson (`CharacterCone`).  `ρ` / `ρ⁻¹` / `trace` delegate to the keystone
(the light-wrapper pattern).  The canonical basis is the keystone's
lower-Kapustin `(m, λ)`; `to_cone_label` / `from_cone_label` implement the
QTCone-sector bijection `((m₁,m₂),(d,d)) ↔ E^{m₁−m₂}·det^{m₂}·w2^d`.
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

from typing import Sequence

from kalgebra import Element, Label
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing, RPowerSeries
from cone_data import ConeData, CrossProductTerm, Cone
from cone_kalgebra import ConeKAlgebra
from pure_u2_qtcone import u2_wilson_fusion


__all__ = ["U2Nf1QTConeData", "U2Nf1QTConeKAlg",
           "E_GEN", "F_GEN", "DET", "DET_INV", "W2", "W2_INV", "WILSON"]

# -- the U(2)+N_f=1 generator labels (lower-Kapustin (m, λ)) -----------------
E_GEN = ((1, 0), (0, 0))      # minuscule monopole E       (conventional)
F_GEN = ((0, -1), (0, 0))     # minuscule monopole F       (conventional)
DET = ((1, 1), (0, 0))        # central 't Hooft det       (conventional — dressed!)
DET_INV = ((-1, -1), (0, 0))  # det⁻¹                      (conventional — dressed!)
W2 = ((0, 0), (1, 1))         # central det-Wilson w2      (TORUS)
W2_INV = ((0, 0), (-1, -1))   # w2⁻¹                       (TORUS)
WILSON = ((0, 0), (1, 0))     # SU(2) fundamental Wilson   (fusing)

# det is matter-dressed at N_f=1 (det·det⁻¹ ≠ 1), so it is a CONVENTIONAL
# gen here — unlike pure U(2), where it was a torus gen.
_CONVENTIONAL = (E_GEN, F_GEN, DET, DET_INV, WILSON)
_TORUS = (W2, W2_INV)


class U2Nf1QTConeData(ConeData):
    """The U(2)+N_f=1 `QTCone` structure (rank-1 central `w2` torus), every
    primitive certified against the keystone `UNNfKAlgebra(2,1)` (retired
    2026-09-19; the algebra passed in is now `U2Nf1FlavourFree`)."""

    def __init__(self, keystone) -> None:
        self._ks = keystone

    def coefficient_ring(self):
        return TrivialZPlusRing()

    # -- generator inventory ---------------------------------------------
    def conventional_gens(self) -> frozenset:
        return frozenset(_CONVENTIONAL)

    def torus_gens(self) -> frozenset:
        return frozenset(_TORUS)

    def mult_gens(self) -> frozenset:
        return frozenset(_CONVENTIONAL) | frozenset(_TORUS)

    def iter_cones(self):
        """The principal `QTCone`: conventional `{E, F, det^{±1}, W}` +
        the surviving rank-1 torus `{w2^{±1}}`."""
        yield Cone(self, self.mult_gens(), torus_gens=self.torus_gens())

    # -- q-commute / cocycle: closed-form (Dirac pairing) -----------------
    @staticmethod
    def _is_qtcone_gen(g) -> bool:
        """A self-computing QTCone-sector generator for U(2)+N_f=1: central
        electric (`λ₁ = λ₂`) **and** central magnetic `m₁+m₂ ≥ 0` (so E /
        det / w2^{±}, but **not** the matter-dressed `det⁻¹` nor F, both of
        which carry `m₁+m₂ < 0` and meet `det` in the meson tower, and not
        the fusing SU(2) Wilson).  These pairwise q-commute."""
        (m1, m2), (l1, l2) = g
        return l1 == l2 and m1 + m2 >= 0

    @staticmethod
    def _dirac(g, h) -> int:
        """The Dirac/DSZ pairing `⟨g,h⟩ = m_g·λ_h − λ_g·m_h` (closed-form)."""
        (gm1, gm2), (gl1, gl2) = g
        (hm1, hm2), (hl1, hl2) = h
        return (gm1 * hl1 + gm2 * hl2) - (gl1 * hm1 + gl2 * hm2)

    def _single_term(self, g, h):
        """Keystone fallback for the matter / Wilson seams."""
        prod = self._ks.multiply(g, h).terms
        if len(prod) != 1:
            return None
        c, lp = next(iter(prod.items()))
        if len(lp._coeffs) != 1:
            return None
        return next(iter(lp._coeffs)), c

    def q_commute(self, g, h) -> bool:
        if g == h:
            return True
        if self._is_qtcone_gen(g) and self._is_qtcone_gen(h):
            return True                       # closed-form (QTCone sector)
        st_gh = self._single_term(g, h)       # keystone fallback
        st_hg = self._single_term(h, g)
        return (st_gh is not None and st_hg is not None
                and st_gh[1] == st_hg[1])

    def cocycle(self, g, h) -> int:
        """`c` with `L_g L_h = q^{c} L_{normal}` (framework convention;
        antisymmetric).  On the QTCone sector the **closed-form Dirac
        pairing** `⟨g,h⟩` (no keystone); matter / Wilson seams fall back to
        the keystone."""
        if g == h:
            return 0
        if self._is_qtcone_gen(g) and self._is_qtcone_gen(h):
            return self._dirac(g, h)          # closed-form
        if not self.q_commute(g, h):
            raise ValueError(f"cocycle: ({g}, {h}) not q-commuting")
        e_gh = self._single_term(g, h)[0]
        assert e_gh == -self._single_term(h, g)[0], (g, h)
        return e_gh

    def cross_product(self, g, h) -> Sequence[CrossProductTerm]:
        """For non-q-commuting gen pairs (the matter / fusing-Wilson
        seams), the keystone product as length-1 canonical words."""
        prod = self._ks.multiply(g, h).terms
        return [(lp, (c,)) for c, lp in prod.items()]

    def canonical_cone_order(self, gens):
        return tuple(sorted(gens, key=lambda x: (x[0], x[1])))

    # -- E·det·w2 monomial bijection (the q-commuting QTCone sector) -------
    def in_qtcone_sector(self, native_label: Label) -> bool:
        """The self-computing sector for U(2)+N_f=1: a *positive*-magnetic
        monomial `E^{m1-m2}·det^{m2}` (`m1 ≥ m2 ≥ 0`, so **no `det⁻¹`** —
        the matter screens the central monopole, and `det·det⁻¹` produces
        the meson tower, leaving the q-commuting sector) dressed by the
        central det-Wilson torus `w2^d` (electric central, `λ1 = λ2`).
        Within this sector all gens q-commute and products close."""
        (m1, m2), (l1, l2) = native_label
        return m1 >= m2 >= 0 and l1 == l2

    def to_cone_label(self, native_label: Label):
        """Decompose a QTCone-sector label `((m1,m2),(d,d))` (m1≥m2≥0) as
        `E^{m1-m2} · det^{m2} · w2^{d}` (det power conventional ≥0; `w2^d`
        the rank-1 torus, inverse gen for negative `d`).  Honest-fail off
        the sector — `det⁻¹` (m2<0) is matter-dressed and an SU(2) Wilson
        (`λ1≠λ2`) is the `CharacterCone` direction; both are delegated."""
        if not self.in_qtcone_sector(native_label):
            raise ValueError(
                f"to_cone_label: {native_label} is outside the q-commuting "
                f"QTCone sector (det⁻¹ matter-dressed, or SU(2) Wilson "
                f"CharacterCone) — handled by the keystone engine")
        (m1, m2), (d, _) = native_label
        cone = self.mult_gens()
        powers = {g: 0 for g in cone}
        if m1 - m2:
            powers[E_GEN] = m1 - m2
        if m2 > 0:
            powers[DET] = m2
        if d > 0:
            powers[W2] = d
        elif d < 0:
            powers[W2_INV] = -d
        return cone, powers

    def from_cone_label(self, gens, powers):
        """Inverse of `to_cone_label` on the QTCone sector."""
        a = powers.get(E_GEN, 0)
        c = powers.get(DET, 0)
        d = powers.get(W2, 0) - powers.get(W2_INV, 0)
        m1, m2 = a + c, c
        return ((m1, m2), (d, d))


class U2Nf1QTConeKAlg(ConeKAlgebra):
    """U(2)+N_f=1 (no flavour) as a `QTCone` `ConeKAlgebra` (delegating
    wrapper over the keystone; canonical basis = lower-Kapustin `(m, λ)`)."""

    _R = TrivialZPlusRing()

    def __init__(self) -> None:
        from u2_nf1_object import U2Nf1FlavourFree   # lazy: the object imports this module
        self._ks = U2Nf1FlavourFree()                  # the general tier's U(2)+1, flavour forgotten (2026-09-19)
        self._cone_data_inst = U2Nf1QTConeData(self._ks)

    def coefficient_ring(self):
        return self._R

    def identity(self) -> Label:
        return self._ks.identity()

    def cone_data(self):
        return self._cone_data_inst

    # -- multiply: self-computing on the QTCone + Wilson sectors ----------
    def multiply(self, a: Label, b: Label) -> Element:
        """**Self-computing** on two closed-form sectors, no keystone call:

        * the q-commuting **QTCone** sector (positive magnetic `E^a·det^c`,
          `c≥0`, dressed by the rank-1 `w2` torus) via
          `cone_data().derived_multiply`;
        * the **Wilson `CharacterCone`** (`m = 0`, both factors) via the
          closed-form U(2) Clebsch `u2_wilson_fusion` — **matter-free**: the
          hyper does not correct the m=0 Wilson fusion (certified == pure
          U(2)), it only screens the monopole sector.

        The remainder — the matter-screened `det⁻¹` and the dyon/meson
        cluster Plücker (monopole–antimonopole annihilation, the rational
        quantum torus) — is taken from the certified keystone engine."""
        cd = self._cone_data_inst
        if cd.in_qtcone_sector(a) and cd.in_qtcone_sector(b):
            return cd.derived_multiply(a, b)
        if a[0] == (0, 0) and b[0] == (0, 0):     # Wilson CharacterCone fusion
            return u2_wilson_fusion(a, b)          # matter-free (m=0 sector)
        return self._ks.multiply(a, b)            # dyon/meson → keystone engine

    def rho(self, label: Label) -> Label:
        return self._ks.rho(label)

    def rho_inverse(self, label: Label) -> Label:
        return self._ks.rho_inverse(label)

    def trace(self, label: Label, K: int = 20) -> RPowerSeries:
        return self._ks.trace(label, K=K)

    def _trace_residual(self, seed_label, K):
        return self._ks.trace(seed_label, K=K)

    # -- convenience ------------------------------------------------------
    def torus_gens(self):
        """The surviving rank-1 central torus `{w2^{±1}}` (the det-Wilson;
        `det` is matter-dressed and no longer invertible)."""
        return self._cone_data_inst.torus_gens()


if __name__ == "__main__":
    A = U2Nf1QTConeKAlg()
    cd = A.cone_data()
    print("U2Nf1QTConeKAlg — QTCone structure (no flavour)\n" + "=" * 48)
    print("coeff ring:", A.coefficient_ring())
    cone = next(iter(cd.iter_cones()))
    print("principal cone is QTCone:", cone.is_quantum_torus())
    print("  conventional:", sorted(cone.conventional_gens()))
    print("  torus (rank-1):", sorted(cone.torus_gens()))
    print("det INVERTIBLE? det·det⁻¹ =", dict(A.multiply(DET, DET_INV).terms),
          "(matter-dressed → conventional)")
    print("w2  INVERTIBLE? w2·w2⁻¹  =", dict(A.multiply(W2, W2_INV).terms),
          "(torus)")
    print("cocycles (certified): w2·E =", cd.cocycle(W2, E_GEN),
          "| w2·det =", cd.cocycle(W2, DET), "| det·E =", cd.cocycle(DET, E_GEN))
    print("meson seam E·F =", dict(A.multiply(E_GEN, F_GEN).terms))
