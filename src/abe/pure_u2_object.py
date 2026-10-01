"""`KAlgebraObject` for **pure U(2) gauge theory**: the abstract `A_𝖖[pure U(2)]` — the *un-projected keystone*
that `pure_su2_object` builds its `abe` leg from (pure SU(2) is the
trace-zero / adjoint-magnetic subalgebra of pure U(2)).

**Trivial flavour.**  Pure gauge theory has no flavour
symmetry, so the coefficient ring is `TrivialZPlusRing` (`= Z`).  Here the
keystone trace is the *honest* U(2) Schur index `1 − 2𝖖² + O(𝖖⁴)` (rank 2:
the SU(2) gauge boson **and** the U(1) photon) — *not* a projected SU(2)
index.  This is the very index the pure-SU(2) `abe` leg inherits and flags
as a "decoupled-photon" caveat; for the U(2) object it is exactly right.

Realizations:

* ``'abe'``  — `PureGAbeKAlgebra(u_n(2))` (until 2026-09-19 the retired
  `PureUNKAlgebra(2)` keystone, measured equal 411/411): the **AbeKAlgebra** leg, the
  pure-U(2) K-theoretic Coulomb-branch algebra on the rational quantum
  torus (`URQTorus`), canonical basis labelled by the lower-Kapustin
  't Hooft–Wilson charge `(m, λ)`.  The canonical, fully-derived,
  honest-trace presentation.

* ``'cone'`` — `PureU2QTConeKAlg`: the **`QTCone` `ConeKAlgebra`**
  presentation (`pure_u2_qtcone.py`).  Pure U(2)'s central `U(1)` — the
  free photon, the very direction that *blocks* a BPS chart (no BPS
  spectrum) — is carried as a `QTCone` **torus direction**: the invertible
  pairs `det^{±1}` (central 't Hooft) and `w2^{±1}` (central det-Wilson),
  spanning the rank-2 central photon torus, with the minuscule monopoles
  `E`/`F` (and the SU(2) Wilson) the conventional gens.  The cone
  cocycles are certified against the keystone; the algebra ops delegate to
  it (the `AbelianizedSU2KAlg` wrapper pattern).  This is the practical
  second presentation — the QTCone mechanism handles the central U(1)
  *algebraically*, where a single BPS chart cannot (see below).

* ``'bps'`` — `BPSKAlgebra(pure_ade.UN_Nf(2, 0))`: the **BPS** realisation
  on the pure-U(2) quiver (rank-4 lattice with the free U(1) photon).
  `multiply` and `ρ` are practical and certified against the keystone via
  the `(m,λ)↔γ` chamber map (`_gamma_u2`/`_label_u2`).  Two documented
  refinements (the `un_nf1` / `pure_su2` pattern of a scoped basket):
  (i) the **monopole×Wilson crossed dyons** need the BPS tropical-charge
  map (the analog of `abelianized_su2_bps_iso`'s `gamma_of`), so they sit
  outside the certified multiply basket; (ii) the bps **trace** is the
  free-photon rank-4 measure residue (impractical; the decoupled-photon
  factor, exactly as in the pure-SU(2) `abe` leg), so trace-equivariance is
  not claimed on the `abe ↔ bps` edge.

* ``'skein'`` — OPEN SLOT (same deferral as `pure_su2_object`).

So the object **wraps the Abe and BPS versions** (abe keystone ↔ bps chart)
plus the QTCone presentation.  The `abe ↔ cone` witness is the identity on
`(m,λ)` (full battery incl. **trace**); the `abe ↔ bps` witness is the
chamber map (unit / round-trip / ρ + clean multiply).  All certified in
the suite in the source repository.
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
from pure_g_abe_kalgebra import PureGAbeKAlgebra
from root_datum import u_n
from pure_u2_qtcone import PureU2QTConeKAlg
from un_bps_chamber import U2TropicalMap


__all__ = ["pure_u2_object"]

_ONE = LaurentPoly.one()


def pure_u2_object(K: int = 12, max_len: int = 2) -> KAlgebraObject:
    """The abstract pure-U(2) algebra as a `KAlgebraObject` wrapping the
    **Abe** keystone, the **BPS** chart, and the **QTCone** presentation
    (`skein` remains an open slot).

    `K` / `max_len` are accepted for source compatibility and ignored (the
    general class builds lazily)."""
    obj = KAlgebraObject("A_q[pure U(2)]")

    abe = PureGAbeKAlgebra(u_n(2))      # the general class; K / max_len ignored since 2026-09-19
    obj.add_realization("abe", abe, {"chart", "multiply-fast",
                                     "f-presentation", "trace-exact"})

    # The QTCone presentation: the central U(1) photon as a quantum-torus
    # direction (det^{±1}/w2^{±1}).  Same canonical (m,λ) basis as the
    # keystone, so the witness is the identity on labels.
    cone = PureU2QTConeKAlg(K=K, max_len=max_len)
    obj.add_realization("cone", cone, {"closed-form", "qtcone",
                                       "multiply-fast", "trace-exact"})

    def _id(label):
        return Element({label: _ONE})

    obj.add_iso("abe", "cone", KAlgebraIso(
        abe, cone, _id, _id,
        name="pure-u2[abe→cone]  (identity on (m,λ); QTCone view)"))

    # The BPS realisation: BPSKAlgebra on UN_Nf(2,0) (the pure-U(2) quiver,
    # rank-4 lattice with the free photon).  multiply / ρ are practical; the
    # abe↔bps witness is the chamber map above.  Certified on
    # unit / round-trip / ρ + the clean multiply sector in the tests; the
    # monopole×Wilson crossed dyons need the BPS tropical-charge refinement
    # (analog of `abelianized_su2_bps_iso`'s gamma_of) and the bps **trace**
    # is the free-photon rank-4 residue (impractical / decoupled-photon, as in
    # the pure-SU(2) abe leg) — so trace-equivariance is not claimed here.
    t0 = pa.UN_Nf(2, 0)
    bps = BPSKAlgebra(pairing=t0.B, node_charges=t0.nodes, spec=t0.spec,
                      cone_witness=getattr(t0, "cone_witness", None),
                      verify="off").shorten_spec()
    obj.add_realization("bps", bps, {"chart", "rg"})

    _tmap = U2TropicalMap(abe, bps)

    def _abe_to_bps(label):
        return Element({_tmap.gamma(label): _ONE})

    def _bps_to_abe(g):
        return Element({_tmap.label(g): _ONE})

    obj.add_iso("abe", "bps", KAlgebraIso(
        abe, bps, _abe_to_bps, _bps_to_abe,
        name="pure-u2[abe→bps]  (UN_Nf(2,0) chart; (m,λ)↔γ chamber map)"))

    return obj


if __name__ == "__main__":
    obj = pure_u2_object()
    print(obj)
    print("keys:", obj.keys())
    abe = obj.realization("abe")
    E, F, DET = ((1, 0), (0, 0)), ((0, -1), (0, 0)), ((1, 1), (0, 0))
    W = ((0, 0), (1, 0))
    print("  E·F   =", dict(abe.multiply(E, F).terms))
    print("  det·W =", dict(abe.multiply(DET, W).terms))
    print("  trace(1) =", {e: c for e, c in abe.trace(abe.identity(), 6).coeffs.items() if c})
