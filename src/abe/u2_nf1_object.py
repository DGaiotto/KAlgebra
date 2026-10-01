"""`KAlgebraObject` for **U(2) + N_f = 1** (no flavour): the abstract
`A_𝖖[U(2)+N_f=1]` holding its **Abelianized** and **BPS** realisations (plus
the QTCone presentation) under one roof, with the certifying `KAlgebraIso`s.

**No flavour.**  At N_f = 1 the genuine flavour group is `SU(N_f) = SU(1) =
trivial`: the hyper's U(1) sits in the U(2) gauge centre, so it is
gauge-centre bookkeeping, not a flavour.  Both realisations are therefore
over `TrivialZPlusRing` (`= Z`):

* the **Abelianized** realisation is the general `(G, N)` tier's U(2)+1 preset
  with its trivial flavour forgotten (`U2Nf1FlavourFree`, below; until
  2026-09-19 the retired `UNNfKAlgebra(2,1)`) and carries no flavour
  coordinate by construction;
* the **BPS** realisation is built *with the flavour charge removed from the
  lattice and node charges* — a `BPSKAlgebra` is defined from a pairing +
  node charges, so dropping the `ker(B)` (gauge-centre U(1)) direction from
  `pure_ade.UN_Nf(2,1)`'s pairing/nodes/spec gives a **non-degenerate**
  rank-4 lattice → `coefficient_ring = TrivialZPlusRing` directly (no
  `forget()` wrapper, a genuine `BPSKAlgebra`).

Realizations:

* ``'abe'``  — `U2Nf1FlavourFree`: `roster('u2-nf1').forget()` with the trivial
  flavour slot stripped, labels `(m, λ)` (the general tier's `(m, e)`).
* ``'bps'``  — `BPSKAlgebra` on the flavour-removed U(2)+N_f=1 quiver
  (3 nodes, rank-4 lattice).  The matter dresses the central monopole
  (`det·det⁻¹` = the meson tower), exactly as the keystone.
* ``'cone'`` — `U2Nf1QTConeKAlg`: the `QTCone` presentation (rank-1 central
  `w2` torus; the matter screens `det`).

Witnesses (`KAlgebraIso`):

* ``abe ↔ bps`` — the `(m,λ) ↔ γ` chamber map (`un_bps_chamber`), certified
  on unit / round-trip / ρ / **trace** + the clean multiply sector.  The
  monopole×Wilson crossed dyons need the BPS tropical-charge refinement
  (analog of `abelianized_su2_bps_iso`'s `gamma_of`), so they sit outside
  the certified multiply basket — the scoped-basket pattern of
  `un_nf1_over_pure_iso` (retired 2026-09-19) and `pure_su2`.
* ``abe ↔ cone`` — the identity on `(m,λ)` (full battery incl. trace).
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

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly
from bps_kalgebra import BPSKAlgebra
import pure_ade as pa

from g_matter_roster import roster
from kalgebra import KAlgebra
from u2_nf1_qtcone import U2Nf1QTConeKAlg
from un_bps_chamber import UNTropicalMap


__all__ = ["u2_nf1_object", "flavour_free_un_nf_bps"]

_ONE = LaurentPoly.one()


class U2Nf1FlavourFree(KAlgebra):
    """U(2)+N_f=1 with its (trivial, gauge-centre) flavour forgotten, on the
    general `(G, N)` tier: `roster('u2-nf1').forget()` re-labelled by the gauge
    part `(m, λ)` alone.

    This is what the retired `UNNfKAlgebra(2,1)` presented — its D8b
    specialization `R(U(1)) → R(SU(1)) = Z` collapsed the det-twisted labels
    onto the untwisted ones, which is exactly `forget()` on the general class
    (measured equal on 406/406 checks across (2,1), (1,3), (2,2) on
    2026-09-19).  The forgotten algebra's labels are `((m, e), (0,))`; the
    object's tropical map wants `(m, e)`, so this wrapper strips the slot."""

    def __init__(self):
        self._G = roster("u2-nf1")
        self._F = self._G.forget()
        self._w0 = self._G.identity()[1]

    def _lift(self, label):
        return (tuple(map(tuple, label)), self._w0)

    @staticmethod
    def _drop(label):
        return label[0]

    def _drop_element(self, x):
        out = {}
        for lab, c in x.terms.items():
            k = self._drop(lab)
            out[k] = out[k] + c if k in out else c
        return Element({k: v for k, v in out.items() if not v.is_zero()})

    def coefficient_ring(self):
        return self._F.coefficient_ring()

    def identity(self):
        return self._drop(self._F.identity())

    def multiply(self, a, b):
        return self._drop_element(self._F.multiply(self._lift(a), self._lift(b)))

    def rho(self, label):
        return self._drop(self._F.rho(self._lift(label)))

    def rho_inverse(self, label):
        return self._drop(self._F.rho_inverse(self._lift(label)))

    def trace(self, label, K: int = 20):
        return self._F.trace(self._lift(label), K=K)

    def _label_section_decompose(self, label):
        return label, self.coefficient_ring().one()

    # -- the AbeKAlgebra acceptance, delegated to the general preset --------

    def certify_canonical(self, label):
        """W1 + W2 acceptance of the lift's chart on the general tier."""
        return self._G.certify_canonical(self._lift(label))

    def chart(self, label):
        return self._G.chart(self._lift(label))


def flavour_free_un_nf_bps(N: int = 2, Nf: int = 1) -> BPSKAlgebra:
    """The U(N)+N_f BPS realisation with the **flavour charge removed from
    the lattice**: drop the `Nf` appended flavour coordinates from
    `pure_ade.UN_Nf(N, Nf)`'s pairing / nodes / spec, leaving the
    non-degenerate rank-`2N` gauge lattice (so `coefficient_ring` is
    `TrivialZPlusRing` — a genuine flavour-free `BPSKAlgebra`)."""
    t = pa.UN_Nf(N, Nf)
    g = 2 * N                                  # gauge-lattice rank
    B = [row[:g] for row in t.B[:g]]
    nodes = [tuple(n[:g]) for n in t.nodes]
    spec = [tuple(s[:g]) for s in t.spec]
    bps = BPSKAlgebra(pairing=B, node_charges=nodes, spec=spec,
                      verify="off").shorten_spec()
    return bps


def u2_nf1_object() -> KAlgebraObject:
    """The abstract U(2)+N_f=1 (no flavour) algebra as a `KAlgebraObject`
    wrapping the Abelianized keystone, the flavour-removed BPS chart, and the
    QTCone presentation — all over trivial flavour."""
    obj = KAlgebraObject("A_q[U(2)+Nf=1]")

    abe = U2Nf1FlavourFree()
    obj.add_realization("abe", abe, {"chart", "f-presentation",
                                     "trace-exact"})

    # BPS with the flavour charge removed from the lattice (genuine
    # BPSKAlgebra, TrivialZPlusRing).
    bps = flavour_free_un_nf_bps(2, 1)
    obj.add_realization("bps", bps, {"chart", "rg", "trace-exact"})

    _tmap = UNTropicalMap(abe, bps, N=2)

    def _abe_to_bps(label):
        return Element({_tmap.gamma(label): _ONE})

    def _bps_to_abe(g):
        return Element({_tmap.label(g): _ONE})

    obj.add_iso("abe", "bps", KAlgebraIso(
        abe, bps, _abe_to_bps, _bps_to_abe,
        name="u2nf1[abe→bps]  (flavour-removed chart; (m,λ)↔γ chamber map)"))

    # The QTCone presentation (rank-1 central w2 torus).
    cone = U2Nf1QTConeKAlg()
    obj.add_realization("cone", cone, {"closed-form", "qtcone",
                                       "multiply-fast", "trace-exact"})

    def _id(label):
        return Element({label: _ONE})

    obj.add_iso("abe", "cone", KAlgebraIso(
        abe, cone, _id, _id,
        name="u2nf1[abe→cone]  (identity on (m,λ); QTCone view)"))

    return obj


if __name__ == "__main__":
    obj = u2_nf1_object()
    print(obj)
    print("keys:", obj.keys())
    bps = obj.realization("bps")
    print("bps coeff_ring:", bps.coefficient_ring(),
          "| isinstance BPSKAlgebra:", isinstance(bps, BPSKAlgebra))
    E, F, CHI, DET = (((1, 0), (0, 0)), ((0, -1), (0, 0)),
                      ((0, 0), (1, 0)), ((1, 1), (0, 0)))
    for lab in [((0, 0), (0, 0)), E, F, CHI, DET]:
        print(f"  abe {lab} → bps {dict(obj.transport(lab, 'abe', 'bps').terms)}")
