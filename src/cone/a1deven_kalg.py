"""
a1deven_kalg.py
===============

`A1DevenKAlg(k)` — the **ungauged** `[A_1, D_{2k+2}]` Argyres–Douglas algebra,
obtained by **ungauging the U(1)** of the spine-free `U1A1DevenConeKAlgebra(k)`.

Construction (mirrors the ungauged A-polygons, generalized to a flavoured base)
------------------------------------------------------------------------------
`U1A1DevenConeKAlgebra(k)` is the U(1)-gauged `[A_1, D_{2k+2}]` with SU(2) matter
flavour (coefficients in `SU2ZPlusRing`).  Its `X_{0,1}` torus generator
`E = ((), 1)` fq-commutes with everything; the magnetic charge `mag(x)` (the
`X_{1,0}` 't Hooft charge) is read off the E-commutator.  The **ungauged**
algebra is the centralizer `Z(E) = {mag = 0}`, with `E` promoted to a **U(1)
flavour fugacity** `z`.  So the ungauged coefficient ring is `SU(2) ⊗ U(1)`.

Unlike the *trivial*-base A-polygons (`HexagonKAlg` etc., where the gauge charge
can stay in the label with scalar coefficients), the D-even base is **SU(2)-
valued**, so `multiply` must lift the gauge (E-) charge of every product term
*into the U(1) fugacity* — leaving an **E-free canonical basis** with `SU(2)⊗U(1)`
coefficients.  That keeps `multiply` and `trace` in the *same* ring, so the
Schur pairing `inner_product` / `verify_orthonormality` are well-typed.

Validation
----------
`trace` reproduces `A1DevenRGKAlgebra(k).trace` **term-for-term** (k=1 through
q^6 — the SU(2)×U(1)-flavoured `[A_1, D_4]` index).  Fully **spine-free**: the
gauged D-even is construction- and runtime-spine-free, and the ungauger touches
no BPS/RG engine.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from zplus_ring import RLaurent
from ungauge_kalgebra import UngaugedKAlgebra


class A1DevenKAlg(UngaugedKAlgebra):
    """Ungauged `[A_1, D_{2k+2}]` (SU(2)×U(1) flavour) — the centralizer of the
    `X_{0,1}` gauge generator in the spine-free `U1A1DevenConeKAlgebra(k)`, with
    the gauge charge lifted into the U(1) fugacity (E-free canonical basis)."""

    def __init__(self, k: int = 1) -> None:
        from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra
        G = U1A1DevenConeKAlgebra(k)
        super().__init__(G, ((), 1), epow=lambda lbl: lbl[1])
        self.k = k

    def identity(self):
        return ((), 0)

    # -- coefficient lift: base RLaurent (SU(2)) ⊗ z^{-f} -> self._R (SU(2)⊗U(1)).
    # The U(1) charge is -f because the inherited ungauged trace obeys
    # Tr_ung((factors, f)) = z^{-f}·Tr_ung((factors, 0)) (the E-power enters the
    # measure-restored z-sum with a minus sign), so stripping E-power f from the
    # label must reintroduce it as z^{-f} for the pairing to stay consistent.
    def _lift(self, lp, f: int) -> RLaurent:
        coeffs = getattr(lp, "coeffs", None)
        if coeffs is None:                          # scalar LaurentPoly fallback
            coeffs = getattr(lp, "_coeffs", {})
            from zplus_ring import RElement
            lifted = {}
            for qp, c in coeffs.items():
                base = self._G.coefficient_ring()
                r = RElement(base, {base.one_basis(): c}) if not hasattr(c, "terms") else c
                lifted[qp] = self._combine_r(r, -f)
            return RLaurent(self._R, lifted)
        return RLaurent(self._R, {qp: self._combine_r(r, -f) for qp, r in coeffs.items()})

    def multiply(self, a, b) -> Element:
        """Centralizer product with the gauge (E-) charge lifted into the U(1)
        fugacity: every mag-0 product term is stripped of its E-power (-> E-free
        canonical label) and that power is absorbed as `z^f` in the `SU(2)⊗U(1)`
        coefficient.  Result lives in the ungauged ring, so it pairs cleanly with
        `trace`."""
        prod = self._G.multiply(a, b)
        out: dict = {}
        for L, lp in prod.terms.items():
            if hasattr(lp, "is_zero") and lp.is_zero():
                continue
            if not self.in_centralizer(L):
                continue                            # drop magnetically-charged terms
            f = self._epow(L)
            L0 = self._e_shift(L, -f)               # strip E-power -> E-free basis
            lifted = self._lift(lp, f)
            out[L0] = (out[L0] + lifted) if L0 in out else lifted
        return Element({L: c for L, c in out.items()
                        if not (hasattr(c, "is_zero") and c.is_zero())})

    def mult_generators(self):
        """The centralizer (mag-0) single-ray generators of the gauged D-even,
        as E-free labels — the ungauged canonical mult-generators.

        Labels are built through the cone bijection `from_cone_label` (NOT the
        raw `(((g, 1),), 0)`), so a torus (gauge `X_{0,1}`) generator maps to its
        proper E-label `((), ±1)` rather than a malformed conventional-ray word.
        The pure-gauge generators (empty ray word) are then dropped: `E` is
        ungauged away (absorbed into the U(1) fugacity), so it is not a generator
        of the ungauged algebra."""
        gens = []
        cd = self._G.cone_data()
        for g in cd.mult_gens():
            lbl = cd.from_cone_label(frozenset([g]), {g: 1})
            factors, _ = lbl
            if not factors:                  # pure-gauge (E) ray -> ungauged away
                continue
            try:
                if self.in_centralizer(lbl):
                    gens.append(lbl)
            except Exception:
                pass
        return gens

    def __repr__(self) -> str:
        return f"A1DevenKAlg(k={self.k})"
