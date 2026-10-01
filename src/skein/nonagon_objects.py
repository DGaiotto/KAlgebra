"""`KAlgebraObject` for the **nonagon** — `A_𝖖([A₁, A₆])`, the A₆
Argyres–Douglas theory (chiral algebra M(2, 9)).

The next **odd** polygon after the pentagon (A₂ = M(2,5)) and the
heptagon (A₄ = M(2,7)); the (2n+1)-gon ↔ A_{2n-2} AD pattern at n = 4.
Built analogously to `kalgebra_object("heptagon")`
(`finite_kalgebras/objects.py`), but standalone here because the
nonagon is **deliberately not a frozen-zoo entry** — its uniform
closed-form presentation `A1A2kKAlg(3)` is complete on its own (the
two-layer trace lands on the four M(2, 9) Andrews–Gordon characters),
so it plays the cone / closed-form role that `cone-frozen` +
`closed-form` split between for the heptagon
(cf. the suite in the source repository,
which certifies the nonagon's trace-uniqueness directly off
`A1A2kKAlg(3)`).

Realizations
------------
* ``'a1a2k'`` — `A1A2kKAlg(3)`: the uniform `[A₁, A_{2k}]` cone
  presentation at k = 3.  Multiply from the per-k base table; the
  two-layer trace reduces (Layer 1 ρ²-cyclicity) to v-tower / single
  chord seeds and substitutes the M(2, 9) characters (Layer 2).
* ``'bps'`` — `BPSKAlgebra` on the **A₆ linear quiver** (the target of
  `a1a2k_bps_iso(3)`); same chamber as the RG flows below.
* ``'rg-u1octagon'`` — `DirectionalSingleNodeRG(A₆, drop one end
  node)`: the nonagon as an **RG flow to the u(1)-gauged octagon**
  (IR auxiliary = the A₅ chamber over the full Z⁶ lattice with the
  dropped direction kept).
* ``'rg-heptagon'`` — `DirectionalSubquiverRG(A₆, drop two adjacent
  nodes)`: the nonagon as an **RG flow to the heptagon** (IR
  auxiliary = the A₄ chamber ⊗ two decoupled directions =
  Heptagon ⊗ QT₂).
* ``'skein-pinned'`` — `SkeinNonagonKAlg`
  (`skein_sphere/skein_nonagon_kalg.py`): the stated-9-gon realization
  extracted from the **pinned** quantum torus (`PinnedPolygon(9)`,
  Γ~ = Z^15).  Odd-marked → no flavour → the clean pentagon story at
  rank 6: the 27 dressed chords `T^(k)_p = lq^{x_k}·F(D_{k,p})·
  NO_p(sides^{-w_{k,p}})` are pin-0, commute with the whole side
  torus (243 checks at construction), and present exactly
  `A1A2kKAlg(3)` on its own labels (D2_p ↔ (1,p), D3_p ↔ (2,p),
  D4_p ↔ (3,p); offset 0).  `multiply` genuinely pinned-side
  (unique-label peel against intrinsic-hinted forward images; each
  peeled coefficient ASSERTED to equal the intrinsic VERBATIM under
  q → lq^{-2} — δ' = 0 everywhere, no normalization table); ρ/trace
  transported from the a1a2k instance it is constructed on.

Witnesses
---------
* a1a2k↔bps — the cone-data-driven mult-gen dictionary
  `a1a2k_bps_iso(3)` (k·(2k+3) = 27 mult-gen images; inverse by
  brute-force charge decomposition).
* skein-pinned↔a1a2k — the identity on labels
  (`SkeinNonagonKAlg.build_iso()`; the skein class is constructed ON
  the registered ``'a1a2k'`` instance, so the witness endpoints are
  the realization instances).  The battery (all 784 generator-pair
  products, 1036 structure constants, verbatim coefficients) lives in
  `skein_sphere/the suite in the source repository.
* rg-*↔bps — the identity on labels (both flows live on the nonagon's
  own Z⁶ BPS charges, same chamber as ``'bps'``); the battery certifies
  that the RG-derived multiply and ρ reproduce the BPS ones.  The
  derived TRACE is exact but intractable in TIME for the directional
  path on Z⁶: since retired the linear `K_joint` prune for the
  sound two-cutoff adaptive shell, the shell re-widens through every RG
  transport order — correct and memory-cheap, but a single rg trace
  (even a bare node at K=4) runs >15 min.  So per-instance
  rg-trace-equivariance is certified generically by the
  directional-machinery suite (`tests/test_directional_subquiver_rg.py`)
  rather than re-run here; the OBJECT's trace is certified through the
  fast closed-form ``a1a2k`` ↔ ``bps`` edge (K=6).

Welding the nonagon to the finite zoo's odd-gon objects (pentagon /
heptagon) is the same tracked follow-up as hexagon ↔ a3.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly
from a1a2k_kalg import A1A2kKAlg
from a1a2k_bps_iso import a1a2k_bps_iso

__all__ = ["nonagon_object"]

_ONE = LaurentPoly.one()


def nonagon_object() -> KAlgebraObject:
    """The abstract nonagon `A_𝖖([A₁, A₆])` (M(2, 9)) as a
    `KAlgebraObject`."""
    obj = KAlgebraObject("A_q[[A1,A6]] (nonagon, M(2,9))")

    # a1a2k (cone / closed-form) ↔ bps (A6 quiver), in one shot from the
    # cone-data-driven factory — its source/target are the registered
    # realizations, so transport and the battery stay instance-consistent.
    iso = a1a2k_bps_iso(3)
    a1a2k, bps = iso.source, iso.target
    obj.add_realization("a1a2k", a1a2k,
                        {"multiply-fast", "trace-exact", "trace-closed-form"})
    obj.add_realization("bps", bps, {"chart", "trace-exact", "rg"})
    obj.add_iso("a1a2k", "bps", iso)

    # The RG flows live on the bps chamber verbatim (read pairing /
    # node charges / spec off the bps so the identity-label witnesses
    # are exact).
    from directional_subquiver_rg import (
        DirectionalSingleNodeRG, DirectionalSubquiverRG)
    P = [list(r) for r in bps.lattice.pairing]
    N = [tuple(g) for g in bps.node_charges]
    S = [tuple(g) for g in bps.spec]

    def _id_iso(src, dst, name):
        return KAlgebraIso(
            src, dst,
            forward_label_map=lambda l: Element({l: _ONE}),
            inverse_label_map=lambda l: Element({l: _ONE}),
            name=name)

    # Nonagon → u(1)-gauged octagon: drop one end node of the A₆ chain.
    rg_oct = DirectionalSingleNodeRG(P, N, S, gamma_drop=0, rg_window=6)
    obj.add_realization("rg-u1octagon", rg_oct, {"rg", "trace-exact"})
    obj.add_iso("rg-u1octagon", "bps",
                _id_iso(rg_oct, bps, "nonagon[rg-u1octagon→bps]"))

    # Nonagon → heptagon: drop two adjacent nodes (IR = Heptagon ⊗ QT₂).
    rg_hept = DirectionalSubquiverRG(P, N, S, drop=[0, 1], rg_window=6)
    obj.add_realization("rg-heptagon", rg_hept, {"rg", "trace-exact"})
    obj.add_iso("rg-heptagon", "bps",
                _id_iso(rg_hept, bps, "nonagon[rg-heptagon→bps]"))

    # The stated-SKEIN realization on the PINNED torus, constructed ON
    # the registered a1a2k instance (so the identity-label witness
    # endpoints are the realization instances — `add_iso` is strict
    # about instance identity).  The geometric certificates (pin-solve,
    # pin-0, 243 side commutations) fire inside the constructor.
    from skein_nonagon_kalg import SkeinNonagonKAlg
    sk = SkeinNonagonKAlg(intrinsic=a1a2k)
    obj.add_realization("skein-pinned", sk, {"geometric"})
    obj.add_iso("skein-pinned", "a1a2k", sk.build_iso())

    return obj
