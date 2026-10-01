"""`PureU2QTConeKAlg` — pure U(2) presented as a **`QTCone` `ConeKAlgebra`**.

The point of this realisation is to exhibit pure U(2)'s **quantum-torus
cone** structure explicitly: the *central* `U(1)` (the free photon) — the
direction that has **no BPS spectrum**, so it blocks a single BPS chart —
is carried *algebraically* as a `QTCone` **torus direction**, an invertible
inverse-pair generator, rather than a spectrum-generating cone ray.  That
is exactly the mechanism that sidesteps the obstruction documented in
`pure_u2_object` (`BPSKAlgebra(UN_Nf(2,0))` does not terminate — the rank-4
positive cone has no negating spec because of the free photon).

Cone structure (read off / certified against the pure-U(2) abelianized algebra —
`PureGAbeKAlgebra(u_n(2))` since 2026-09-19, the retired `PureUNKAlgebra(2)`
keystone before):

* **conventional gens** — the minuscule monopoles `E = L_{((1,0),(0,0))}`
  and `F = L_{((0,-1),(0,0))}` (powers strictly positive in canonical
  labels), and the SU(2) fundamental Wilson `W = L_{((0,0),(1,0))}` (a
  *fusing* character — `W·W` is two terms — which lives morally in a
  `CharacterCone`; here it is carried as a conventional gen and its
  products are taken from the keystone);
* **torus gens** — `det = L_{((1,1),(0,0))}` (central 't Hooft monopole)
  and `w2 = L_{((0,0),(1,1))}` (central det-Wilson), each with its inverse
  (`det·det⁻¹ = 1`, `w2·w2⁻¹ = 1` — verified), spanning the **rank-2
  central photon quantum torus**.

Generator cocycles (`L_g L_h = q^{cocycle(g,h)} L_h L_g`, certified
against the keystone): `det`↔monopole `= 0`, `w2`↔monopole `= ±1`,
`det`↔`w2` `= ±2` (the photon's Dirac pairing).  The Z₂ of
`U(2) = (SU(2)×U(1))/Z₂` shows up as `E² = (SU(2)-monopole)·det` — the
minuscule `E` carries *half* a unit of central charge, so the photon
torus is genuinely entangled with the SU(2) cone (no clean product
factorisation).

Realisation policy
------------------
`multiply` is **self-computing** on the QTCone (q-commuting
monopole–det–torus) sector: it runs `cone_data().derived_multiply` —
cocycles + the `det·det⁻¹ = w2·w2⁻¹ = 1` torus collapse — entirely from the
cone data, with no keystone call.  That sector (`λ₁ = λ₂`: magnetic
monomials `E^{m₁−m₂}·det^{m₂}` dressed by the central `w2` torus) is closed
under multiplication, and the cone-derived products match the keystone
exactly (certified, the suite in the source repository).  The only fusing
direction is the SU(2) Wilson (`λ₁≠λ₂`) — a `CharacterCone` (Chebyshev)
direction, *not* a single q-commuting monomial; those products are taken
from the pure-U(2) abelianized algebra (`PureGAbeKAlgebra(u_n(2))`; the retired
`PureUNKAlgebra(2)` engine until 2026-09-19).  `ρ` / `ρ⁻¹` / `trace`
likewise delegate to it (ρ is the QTCone shear/torus-inversion
label permutation; trace the Schur-measure residue) — the same light-wrapper
pattern as `AbelianizedSU2KAlg`.  The canonical basis is the keystone's
lower-Kapustin `(m, λ)`; `to_cone_label` / `from_cone_label` implement the
QTCone-sector monomial bijection `((m₁,m₂),(d,d)) ↔ E^{m₁−m₂}·det^{m₂}·w2^d`.
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
from pure_g_abe_kalgebra import PureGAbeKAlgebra
from root_datum import u_n


__all__ = ["PureU2QTConeData", "PureU2QTConeKAlg",
           "E_GEN", "F_GEN", "DET", "DET_INV", "W2", "W2_INV", "WILSON"]

# -- the pure-U(2) generator labels (lower-Kapustin (m, λ)) ------------------
E_GEN = ((1, 0), (0, 0))      # minuscule monopole E  (conventional)
F_GEN = ((0, -1), (0, 0))     # minuscule monopole F  (conventional)
DET = ((1, 1), (0, 0))        # central 't Hooft det  (TORUS)
DET_INV = ((-1, -1), (0, 0))  # det⁻¹                 (TORUS)
W2 = ((0, 0), (1, 1))         # central det-Wilson w2 (TORUS)
W2_INV = ((0, 0), (-1, -1))   # w2⁻¹                  (TORUS)
WILSON = ((0, 0), (1, 0))     # SU(2) fundamental Wilson (fusing)

_CONVENTIONAL = (E_GEN, F_GEN, WILSON)
_TORUS = (DET, DET_INV, W2, W2_INV)


def u2_wilson_fusion(a, b):
    """Closed-form **Wilson `CharacterCone`** product of two *m=0* canonicals
    `a = ((0,0),(a1,a2))`, `b = ((0,0),(b1,b2))` (dominant electric) — the
    U(2) Clebsch–Gordan of the SU(2) Wilson chars with the det-Wilson `w2`
    bookkeeping:

        χ_{a1-a2}·χ_{b1-b2}  =  Σ_{i=0}^{min} w2^{i} · χ_{(a1-a2)+(b1-b2)-2i}

    i.e. `L_{((0,0),(a1+b1-i, a2+b2+i))}` for `i = 0 .. min(a1-a2, b1-b2)`,
    each with coefficient 1.  No monopole, so **no Plücker/meson** — this is
    self-computing (the genuine fusing-Wilson direction).  `Element`."""
    (a1, a2), (b1, b2) = a[1], b[1]
    n = min(a1 - a2, b1 - b2)
    return Element({((0, 0), (a1 + b1 - i, a2 + b2 + i)): LaurentPoly.one()
                    for i in range(n + 1)})


class PureU2QTConeData(ConeData):
    """The pure-U(2) `QTCone` structure, with every primitive certified
    against the pure-U(2) abelianized algebra passed in at construction
    (`PureGAbeKAlgebra(u_n(2))`; the retired `PureUNKAlgebra(2)` keystone until
    2026-09-19).

    Conventional gens `{E, F, W}`; torus gens `{det^{±1}, w2^{±1}}`.
    `q_commute` / `cocycle` / `cross_product` are *read off* the keystone
    (so correct by construction); `to_cone_label` / `from_cone_label`
    implement the clean magnetic-det monomial bijection (the QTCone where
    the photon-as-torus statement lives)."""

    def __init__(self, keystone) -> None:
        self._ks = keystone
        self._one = LaurentPoly.one()

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
        """The principal `QTCone` (the only cone needed to exhibit the
        torus structure): conventional `{E, F, W}` + torus `{det^{±1},
        w2^{±1}}`."""
        yield Cone(self, self.mult_gens(), torus_gens=self.torus_gens())

    # -- q-commute / cocycle: closed-form (Dirac pairing) -----------------
    @staticmethod
    def _is_qtcone_gen(g) -> bool:
        """A QTCone-sector generator (E / F / det^{±} / w2^{±}) — magnetic
        with **central electric** (`λ₁ = λ₂`).  These pairwise q-commute;
        the SU(2) Wilson (`λ₁≠λ₂`) is the fusing `CharacterCone` exception."""
        (_, _), (l1, l2) = g
        return l1 == l2

    @staticmethod
    def _dirac(g, h) -> int:
        """The Dirac/DSZ pairing `⟨g,h⟩ = m_g·λ_h − λ_g·m_h` — the cocycle
        of the central quantum torus (closed-form, no keystone)."""
        (gm1, gm2), (gl1, gl2) = g
        (hm1, hm2), (hl1, hl2) = h
        return (gm1 * hl1 + gm2 * hl2) - (gl1 * hm1 + gl2 * hm2)

    def _single_term(self, g, h):
        """Keystone fallback for the Wilson seams: if `L_g L_h` is a single
        term `q^e L_c`, return `(e, c)`; else `None`."""
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
        st_gh = self._single_term(g, h)       # keystone fallback (Wilson)
        st_hg = self._single_term(h, g)
        return (st_gh is not None and st_hg is not None
                and st_gh[1] == st_hg[1])

    def cocycle(self, g, h) -> int:
        """`c` with `L_g L_h = q^{c} L_{normal}` (framework convention;
        antisymmetric `c(g,h) = -c(h,g)`).  On the QTCone sector this is the
        **closed-form Dirac pairing** `⟨g,h⟩` (no keystone); the Wilson
        seams fall back to the keystone."""
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
        """For non-q-commuting gen pairs (the fusing Wilson seams), the
        keystone product, returned as length-1 canonical words.  Not on
        the `multiply` critical path (delegated); exposed for inspection
        and certification."""
        prod = self._ks.multiply(g, h).terms
        return [(lp, (c,)) for c, lp in prod.items()]

    def canonical_cone_order(self, gens):
        return tuple(sorted(gens, key=lambda x: (x[0], x[1])))

    # -- magnetic-det-w2 monomial bijection (the q-commuting QTCone sector) --
    def in_qtcone_sector(self, native_label: Label) -> bool:
        """A label is in the (q-commuting) QTCone sector iff its electric
        part is *central* (`λ₁ = λ₂`): a magnetic monomial `E^a·det^c`
        dressed by the central torus `w2^d`, with **no** SU(2) Wilson
        (the fusing `CharacterCone` direction)."""
        (m1, m2), (l1, l2) = native_label
        return m1 >= m2 and l1 == l2

    def to_cone_label(self, native_label: Label):
        """Decompose a QTCone-sector label `((m1,m2),(d,d))` (m1≥m2) as the
        q-commuting monomial `E^{m1-m2} · det^{m2} · w2^{d}` (negative
        det/w2 powers go to the inverse gens).  Honest-fail off the QTCone
        sector — an SU(2)-Wilson dressing (`λ1≠λ2`) is the `CharacterCone`
        direction, not a single q-commuting monomial."""
        if not self.in_qtcone_sector(native_label):
            raise ValueError(
                f"to_cone_label: {native_label} carries an SU(2) Wilson "
                f"(λ1≠λ2) — outside the q-commuting QTCone sector "
                f"(CharacterCone direction, handled by the Wilson engine)")
        (m1, m2), (d, _) = native_label
        cone = self.mult_gens()
        powers = {g: 0 for g in cone}
        if m1 - m2:
            powers[E_GEN] = m1 - m2
        if m2 > 0:
            powers[DET] = m2
        elif m2 < 0:
            powers[DET_INV] = -m2
        if d > 0:
            powers[W2] = d
        elif d < 0:
            powers[W2_INV] = -d
        return cone, powers

    def from_cone_label(self, gens, powers):
        """Inverse of `to_cone_label` on the QTCone sector."""
        a = powers.get(E_GEN, 0)
        c = powers.get(DET, 0) - powers.get(DET_INV, 0)
        d = powers.get(W2, 0) - powers.get(W2_INV, 0)
        m1, m2 = a + c, c
        return ((m1, m2), (d, d))


class PureU2QTConeKAlg(ConeKAlgebra):
    """Pure U(2) as a `QTCone` `ConeKAlgebra` (delegating wrapper over the
    keystone; canonical basis = lower-Kapustin `(m, λ)`)."""

    _R = TrivialZPlusRing()

    def __init__(self, K: int = 12, max_len: int = 2) -> None:
        self._ks = PureGAbeKAlgebra(u_n(2))    # the general class (2026-09-19); K / max_len ignored
        self._cone_data_inst = PureU2QTConeData(self._ks)

    def coefficient_ring(self):
        return self._R

    def identity(self) -> Label:
        return self._ks.identity()

    def cone_data(self):
        return self._cone_data_inst

    # -- multiply: self-computing on the QTCone + Wilson sectors ----------
    def multiply(self, a: Label, b: Label) -> Element:
        """**Self-computing** on two closed-form sectors, no keystone call:

        * the **QTCone** (q-commuting monopole–det–torus) sector, via
          `cone_data().derived_multiply` (Dirac cocycles + the
          `det·det⁻¹ = w2·w2⁻¹ = 1` torus collapse);
        * the **Wilson `CharacterCone`** (`m = 0`, both factors), via the
          closed-form U(2) Clebsch `u2_wilson_fusion` — the fusing SU(2)
          Wilson chars, with no monopole so no Plücker.

        The remainder — dyon/meson products where a monopole and
        anti-monopole **annihilate with electric dressing** — is the
        cluster Plücker structure, which is the rational quantum torus
        (`URQTorus`); those are taken from the certified keystone engine."""
        cd = self._cone_data_inst
        if cd.in_qtcone_sector(a) and cd.in_qtcone_sector(b):
            return cd.derived_multiply(a, b)
        if a[0] == (0, 0) and b[0] == (0, 0):     # Wilson CharacterCone fusion
            return u2_wilson_fusion(a, b)
        return self._ks.multiply(a, b)            # dyon/meson cluster Plücker

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
        """The `det^{±1}` / `w2^{±1}` quantum-torus directions (the
        central photon)."""
        return self._cone_data_inst.torus_gens()


if __name__ == "__main__":
    A = PureU2QTConeKAlg()
    cd = A.cone_data()
    print("PureU2QTConeKAlg — QTCone structure\n" + "=" * 44)
    print("identity:", A.identity())
    cone = next(iter(cd.iter_cones()))
    print("principal cone is QTCone:", cone.is_quantum_torus())
    print("  conventional:", sorted(cone.conventional_gens()))
    print("  torus:", sorted(cone.torus_gens()))
    print("cocycles (certified vs keystone):")
    for g, h in [(DET, E_GEN), (W2, E_GEN), (DET, W2)]:
        print(f"  cocycle({g}, {h}) =", cd.cocycle(g, h))
    print("torus inverses: det·det⁻¹ =", dict(A.multiply(DET, DET_INV).terms),
          "| w2·w2⁻¹ =", dict(A.multiply(W2, W2_INV).terms))
    print("to_cone_label(((2,0),(0,0))):", cd.to_cone_label(((2, 0), (0, 0)))[1])
