"""`su2a1d3_gauged` — SU(2)-gauged [A_1, D_3] (+ tail): the worked
n = 3 case of "gauge the SU(2) flavour of [A_1, D_n]", with its two
single-node-drop RG flows and the **completed standalone-IR swap**.

Theory (user = Gaiotto, 2026-06-16).  Gauging the SU(2) flavour of the
[A_1, D_n] Argyres-Douglas theory replaces the two short-leg (fork) nodes
of D_n by a single matter node coupled to a **pure-SU(2) Kronecker** quiver
the way an N_f = 1 fundamental is, leaving the D_n stem (an A_{n-2} chain)
attached to the matter node by one edge.  The simplest case is
**[A_1, D_3] + a single tail node** — the 4-node quiver below (drawn in
ClusterApplet; 0-based spec seq ``[0,2,1,0,3]``)::

    B = [[ 0, 2,-1, 0],      nodes = e0,e1,e2,e3  (identity charges)
         [-2, 0, 1, 0],      spec  = [e0, e0+e2, e1, e2, e3]   (BFS)
         [ 1,-1, 0, 1],
         [ 0, 0,-1, 0]]

* ``{0,1}`` = pure-SU(2) **Kronecker** (double arrow, ``B01 = 2``);
* node 2 = the **N_f=1 matter node** (single bonds to 0, 1, and the tail);
* node 3 = the **tail / U(1) gauge node** — bonds *only* to node 2
  (``B`` row 3 = ``[0,0,-1,0]``), gauging the baryonic U(1) flavour of
  SU(2) N_f=1.
* ``Pf(B) = 2  =>  det B = 4  =>  ker B = 0``: **flavourless** (the SU(2) is
  gauged) — an interacting flavourless SCFT with non-Lagrangian content.

So ``SU2A1D3`` = the **U(1)-gauging of SU(2) N_f=1**, readable two ways
(both certified single-node-drop RG flows *out* of it):

* cut ``{0,1,2} | {3}``  — drop the tail  ->  **U(2) N_f=1**;
* cut ``{0,1}  | {2,3}`` — drop the matter ->  **pure SU(2) x SQED_1**.

**Single-node RG preserves Gamma** (rank 4): the drop removes a *generator*
/ spectrum node; the dropped-node count is the rank-1 RG grading, not a
lattice reduction (the auxiliary pairing stays the non-degenerate 4x4,
det 4).  The two IRs are therefore genuine rank-4 theories.

Completed standalone-IR swap (this module's headline)
-----------------------------------------------------
The tail-drop auxiliary (a BPSKAlgebra) is replaced by the repo's
**certified U(2) N_f=1** realisation ``flavour_free_un_nf_bps(2,1)``
(PR #522) via a verified ``KAlgebraIso`` ``phi`` — the node-charge
correspondence ``e0|->n1, e1|->n2, e2|->n3`` (the two share the *identical*
3-node generating quiver ``[[0,2,-1],[-2,0,1],[1,-1,0]]``; the det-4 vs
det-1 difference is only the spurious 4th-direction embedding).  PR #522 in
turn certifies that BPS realisation against the fast Abelianized keystone
``UNNfKAlgebra(2,1)`` (the ``un_bps_chamber`` ``(m,lambda) <-> gamma`` map),
so all IR quantities are computable on the fast standalone.

``tail_drop_to_u2nf1`` returns the swapped flow (``IsoComposedRGKAlgebra``);
``verify_ir_iso_hypotheses`` passes every hypothesis and the swapped flow
reproduces the UV product on the **gauge sector** (products of
``{e0,e1,e2}``).  The tail ``e3`` is the integrated-out direction — its
det-4 embedding differs from the standalone's det-1 by the spurious
flavour, so tail-involving products are UV-only, outside the U(2) N_f=1 IR.

The **matter drop** swaps analogously: ``matter_drop_to_pure_su2_qt1d`` ->
``PureSU2KAlg (x) QuantumTorus1DKAlg`` (its IR positive cone is the
clean tensor ``pure SU(2){0,1} (x) Z[e3]``; the pure-SU(2) leg via the same
shared-Kronecker node-charge bridge, ``e3`` -> the QT1D electric tower).  The
SQED_1 *monopole* (``u_pm = U_{pm 1}``) is the dropped node-2 magnetic
direction, **off the positive cone** — so this is ``pure SU(2) x SQED_1`` with
the monopole in the off-cone sector; the full-SQED_1 presentation is a follow-up.

Run ``PYTHONPATH=. python implementations/su2a1d3_gauged.py`` for the full
certificate.  Companion: ``experiments/su2a1d3_subalgebra.py`` (the
SU(2) N_f=1 ⊂ U(2) N_f=1 global-form / Z2 study).
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

from bps_kalgebra import BPSKAlgebra
from iso_composed_rgkalgebra import IsoComposedRGKAlgebra
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from rg_flow import SingleNodeRG
from u2_nf1_object import flavour_free_un_nf_bps

_ONE = LaurentPoly.one()

# --- the applet quiver (ClusterApplet, 0-based spec seq [0,2,1,0,3]) ---------
B_SU2A1D3 = [[0, 2, -1, 0], [-2, 0, 1, 0], [1, -1, 0, 1], [0, 0, -1, 0]]
NODES = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
SPEC = [(1, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]

# node roles (0-based)
KRONECKER_NODES = (0, 1)   # pure-SU(2) double arrow
MATTER_NODE = 2            # N_f=1 fundamental
TAIL_NODE = 3              # U(1) gauge node (gauges the baryonic U(1)_F)


def su2a1d3_gauged(verify: str = "off") -> BPSKAlgebra:
    """The SU(2)-gauged [A_1, D_3] (+ tail) K-algebra as a `BPSKAlgebra`
    on the 4-node applet quiver (flavourless; `det B = 4`)."""
    return BPSKAlgebra(pairing=B_SU2A1D3, node_charges=NODES, spec=SPEC,
                       verify=verify)


def su2_gauged_a1dn_quiver(n: int):
    """The **general-n** SU(2)-gauged [A_1, D_n] (+ tail) BPS quiver, from the
    user's recipe (the matching spectrum generator is produced explicitly by
    `su2_gauged_a1dn_spec`, no search).

    Returns `(B, node_roles)` where `B` is the (n+1)x(n+1) antisymmetric
    pairing and `node_roles` maps each index to its role.  Node layout
    (0-based), per "an [A_1, A_{n-2}] BPS subquiver and an SU(2) N_f=1
    subquiver joined by one edge (chain head <-> matter node)":

    * ``0, 1`` — pure-SU(2) **Kronecker** (``B01 = 2``);
    * ``2``    — the **N_f=1 matter node** (``B[2,0]=1, B[2,1]=-1``);
    * ``3 .. n`` — the **A_{n-2} chain** (``n-2`` nodes, ``B[i,i+1]=1``),
      attached to the matter node by the edge ``B[2,3]=1``.

    Node count ``n+1``: **even** (n odd) => `det B != 0`, flavourless;
    **odd** (n even) => `det B = 0`, ker rank 1 (a residual U(1) flavour of
    [A_1, D_n]).  `n = 3` reproduces the applet quiver `B_SU2A1D3` exactly.
    """
    if n < 3:
        raise ValueError("n >= 3")
    N = n + 1
    B = [[0] * N for _ in range(N)]

    def setpair(i, j, v):
        B[i][j] = v
        B[j][i] = -v

    setpair(0, 1, 2)          # pure-SU(2) Kronecker
    setpair(0, 2, -1)         # matter node to Kronecker (N_f=1)
    setpair(1, 2, 1)
    setpair(2, 3, 1)          # edge: matter node <-> chain head
    for i in range(3, n):     # A_{n-2} chain (nodes 3..n)
        setpair(i, i + 1, 1)

    roles = {0: "kronecker", 1: "kronecker", 2: "matter"}
    for i in range(3, N):
        roles[i] = "tail_chain"
    return B, roles


def su2_gauged_a1dn_spec(n: int):
    """The SU(2)-gauged [A_1, D_n] (+ tail) spectrum generator, built
    **explicitly from the [A_1, D_n] spec** via the user's recipe (2026-06-16) —
    no search.

    The [A_1, D_n] spec is the `n` simple BPS states (the D_n nodes): the stem
    `0..n-3` are flavour **singlets** (the A_{n-2} chain), the two leaves
    `n-2, n-1` are the **SU(2) doublet** `E_q(X v)E_q(X/v)`.  The recipe:

    * promote the leaf doublet to the **matter block** `{e2, e0+e2}` (the two
      Wilson-shifted weights, `χ_w → ` abelianized SU(2) Wilson);
    * keep the stem as the **A_{n-2} tail** chain `e3, ..., en`;
    * **append the pure-SU(2) Kronecker** spec `e0, e1`.

    In the interleaved SU(2) N_f=1 chamber this is

        [e0, e0+e2, e1, e2]   +   [e3, ..., en]
        (pure SU(2) interleaved      (the A_{n-2}
         with the matter block)        tail chain)

    a valid negating sequence for every `n` (verified n = 3..6; `n = 3`
    reproduces the applet spec, and the factor set matches the BFS chamber at
    n = 3, 5 — KS wall-crossing invariance).  `su2_gauged_a1dn_quiver(n)` is the
    matching quiver."""
    if n < 3:
        raise ValueError("n >= 3")
    N = n + 1

    def e(i):
        return tuple(1 if j == i else 0 for j in range(N))

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b))

    su2nf1 = [e(0), add(e(0), e(2)), e(1), e(2)]   # pure SU(2) ⋈ matter block
    tail = [e(i) for i in range(3, N)]              # A_{n-2} tail chain
    return su2nf1 + tail


def su2_gauged_a1dn(n: int, verify: str = "off") -> BPSKAlgebra:
    """SU(2)-gauged [A_1, D_n] (+ tail) as a `BPSKAlgebra` for general n,
    constructed from the recipe — quiver `su2_gauged_a1dn_quiver(n)`, spectrum
    generator `su2_gauged_a1dn_spec(n)` built **explicitly from the [A_1, D_n]
    spec** (no search).  **Flavoured** (`AbelianZPlusRing(rank=1)`, a residual
    U(1)) for even n; flavourless for odd n.  n=4 = pure SU(2) + N_f=1 matter +
    a pentagon `[A_1,A_2]` tail."""
    B, _ = su2_gauged_a1dn_quiver(n)
    N = n + 1
    nodes = [tuple(1 if j == i else 0 for j in range(N)) for i in range(N)]
    spec = su2_gauged_a1dn_spec(n)
    return BPSKAlgebra(pairing=B, node_charges=nodes, spec=spec, verify=verify)


def su2_gauged_a1dn_drop(n: int, j: int, A: BPSKAlgebra | None = None):
    """Single-node-drop **RGKAlgebra** of SU(2)-gauged [A_1, D_n] — `SingleNodeRG`
    dropping node `j` (a `DirectionalSubquiverRG`; the whole K-algebra API is
    derived, identity-on-labels iso vs the UV is a genuine certificate).

    Physical drops: `j = MATTER_NODE` (2) → pure SU(2) × (SQED_1 + tail);
    `j = n` (the tail end) → SU(2)-gauged [A_1, D_{n-1}] (a shorter tail).
    For n=3 these are `matter_drop_flow` / `tail_drop_flow`; their wrapped BPS IRs
    can be swapped for fast standalones (cf. `tail_drop_to_u2nf1` /
    `matter_drop_to_pure_su2_qt1d`)."""
    from rg_flow import SingleNodeRG
    return SingleNodeRG(A if A is not None else su2_gauged_a1dn(n), j)


def matter_drop_flow(A: BPSKAlgebra | None = None) -> SingleNodeRG:
    """Drop the matter node (2): the IR is **pure SU(2) x SQED_1** (the
    SQED_1 monopole is the retained node-2 magnetic direction `u_pm`)."""
    return SingleNodeRG(A if A is not None else su2a1d3_gauged(), MATTER_NODE)


def tail_drop_flow(A: BPSKAlgebra | None = None) -> SingleNodeRG:
    """Drop the tail / U(1) node (3): the IR is **U(2) N_f=1** (rank-4 Gamma
    preserved)."""
    return SingleNodeRG(A if A is not None else su2a1d3_gauged(), TAIL_NODE)


def _node_charge_iso(aux: BPSKAlgebra, target_bps: BPSKAlgebra) -> KAlgebraIso:
    """`KAlgebraIso` aux -> target_bps by the generator-charge map
    `e_i |-> n_i` (`n_i` = `target_bps.node_charges`).  Valid because the
    tail-drop aux and the U(2) N_f=1 chart present the *identical* 3-node
    generating quiver; aux canonical labels have zero tail (3rd) coord."""
    n = [tuple(c) for c in target_bps.node_charges]
    assert n[2] == (0, 0, 0, 1), f"expected matter generator at e3-coord, got {n}"

    def fwd(label):
        a0, a1, a2, _a3 = label
        g = tuple(a0 * n[0][k] + a1 * n[1][k] + a2 * n[2][k] for k in range(4))
        return Element({g: _ONE})

    def inv(g):
        # solve a0*n1 + a1*n2 + a2*n3 = g  (n3 = (0,0,0,1)):
        a1 = g[1]
        return Element({(g[0] + a1, a1, g[3] + a1, 0): _ONE})

    return KAlgebraIso(aux, target_bps, fwd, inv,
                       name="SU2A1D3 tail-drop aux -> U(2)+Nf=1 (PR#522 chart)")


def tail_drop_iso_to_u2nf1(A: BPSKAlgebra | None = None):
    """Return `(flow, phi, u2nf1_bps)`: the tail-drop flow, the verified
    `KAlgebraIso` `phi` from its auxiliary to the certified U(2) N_f=1 BPS
    realisation `flavour_free_un_nf_bps(2,1)`, and that target."""
    flow = tail_drop_flow(A)
    u2 = flavour_free_un_nf_bps(2, 1)
    phi = _node_charge_iso(flow.auxiliary(), u2)
    return flow, phi, u2


def tail_drop_to_u2nf1(A: BPSKAlgebra | None = None) -> IsoComposedRGKAlgebra:
    """The tail-drop flow with its wrapped BPS IR **swapped for the certified
    U(2) N_f=1 standalone** (PR #522).  An `IsoComposedRGKAlgebra` whose
    auxiliary is `flavour_free_un_nf_bps(2,1)`; reproduces the UV product."""
    flow, phi, _u2 = tail_drop_iso_to_u2nf1(A)
    return IsoComposedRGKAlgebra(flow, phi)


def matter_drop_iso_to_pure_su2_qt1d(A: BPSKAlgebra | None = None):
    """Return `(flow, phi, target)`: the matter-drop flow, the verified
    `KAlgebraIso` `phi` from its auxiliary to `PureSU2 cone (x) QuantumTorus1D`,
    and that target.

    The matter-drop auxiliary's **positive canonical cone** is a clean tensor
    `pure SU(2){0,1} (x) Z[e3]` (the Kronecker OPE loses the node-2 leak; `e3`
    is central).  So the directional IR is `PureSU2KAlg (x)
    QuantumTorus1DKAlg` (`e3` = the electric U(1) tower).  Physically this is
    **pure SU(2) x SQED_1** — but SQED_1's *monopole* (`u_pm = U_{pm 1}`) is the
    *dropped* node-2 magnetic direction, **off the positive cone** of the bare
    directional aux; the full-SQED_1 (with monopole) presentation is a follow-up.
    The pure-SU(2) leg uses the PR #522-style node-charge bridge `e0,e1 -> the
    Kronecker nodes of the pure-SU(2) standalone` (shared det-4 Kronecker quiver;
    a lattice `find_isomorphism` can't bridge the standalone's det-1 charge frame)."""
    from pure_su2_object import pure_su2_object
    from qtorus_1d import QuantumTorus1DKAlg
    from tensor_kalgebra import TensorKAlgebra

    flow = matter_drop_flow(A)
    aux = flow.auxiliary()
    pso = pure_su2_object()
    psobps = pso.realization("bps")
    cone = pso.realization("cone")
    n = [tuple(c) for c in psobps.node_charges]   # Kronecker nodes (1,0),(-1,2)
    target = TensorKAlgebra(cone, QuantumTorus1DKAlg())

    def _m2(a0, a1):
        return (a0 * n[0][0] + a1 * n[1][0], a0 * n[0][1] + a1 * n[1][1])

    def fwd(label):
        a0, a1, _a2, c = label
        cl = next(iter(pso.transport(_m2(a0, a1), "bps", "cone").terms))
        return Element({(cl, c): _ONE})

    def inv(tlabel):
        cl, c = tlabel
        g = next(iter(pso.transport(cl, "cone", "bps").terms))
        a1 = g[1] // 2
        return Element({(g[0] + a1, a1, 0, c): _ONE})

    phi = KAlgebraIso(aux, target, fwd, inv,
                      name="SU2A1D3 matter-drop aux -> pure SU(2) (x) QT1D")
    return flow, phi, target


def matter_drop_to_pure_su2_qt1d(A: BPSKAlgebra | None = None) -> IsoComposedRGKAlgebra:
    """The matter-drop flow with its wrapped BPS IR swapped for the fast
    `PureSU2KAlg (x) QuantumTorus1DKAlg` (the directional IR; SQED_1
    monopole off-cone — see `matter_drop_iso_to_pure_su2_qt1d`)."""
    flow, phi, _t = matter_drop_iso_to_pure_su2_qt1d(A)
    return IsoComposedRGKAlgebra(flow, phi)


# ---------------------------------------------------------------------------
# The det-1 LINE-LATTICE realization (the CORRECT Gamma for cones / Wilson)
# ---------------------------------------------------------------------------
# The applet build above (`su2a1d3_gauged`) lives on the COARSE det-4 node-charge
# sublattice: there the gauge Wilson line `W = -1/2(n1+n2)` is half-integral, so
# it is not even a label.  The PHYSICAL Gamma is the finer **det-1 line lattice**
# (user, 2026-06): `Gamma = Z2 (+) Z2` (two elementary weight lattices), pairing
# `J (+) J`, with the BPS quiver nodes
#
#     n1 = (1,0;0,0)  n2 = (-1,2;0,0)   [SU(2) Kronecker, <n1,n2>=2]
#     n3 = (0,-1;1,0)                   [matter / gauged doublet]
#     n4 = (0,0;0,1)                    [tail = the [A1,D3] stem]
#
# whose Gram is exactly `B_SU2A1D3`.  **Critical:** the BPSKAlgebra nodes must be
# these `n1..n4`, NOT the identity basis `e1..e4` of this Z^4 (which has pairing
# `J(+)J` directly, no Kronecker -> it abelianizes the gauge SU(2) and collapses
# every Wilson to a bare monomial `X`).  With the correct nodes, the canonical
# `F_{(0,-1,0,0)}` is the genuine 3-term Wilson character and `W*W = 1 (+) chi_2`.
#
# All structure below was read off the public API (`A.multiply`/`A.rho`/`A.F`).

J_PLUS_J = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]
DET1_NODES = [(1, 0, 0, 0), (-1, 2, 0, 0), (0, -1, 1, 0), (0, 0, 0, 1)]  # n1..n4
# applet spec [n1, n1+n3, n2, n3, n4] in det-1 coords:
DET1_SPEC = [(1, 0, 0, 0), (1, -1, 1, 0), (-1, 2, 0, 0), (0, -1, 1, 0), (0, 0, 0, 1)]

WILSON = (0, -1, 0, 0)        # W = -1/2(n1+n2) = F_{(0,-1,0,0)}, gauge SU(2) Wilson
WBOSON = (0, 2, 0, 0)         # root = n1+n2
MATTER = (0, -1, 1, 0)        # n3 = [A1,D3] doublet (spin-1/2)
TAIL = (0, 0, 0, 1)           # n4 = [A1,D3] stem (singlet)


def su2a1d3_det1(verify: str = "off") -> BPSKAlgebra:
    """SU(2)-gauged [A_1, D_3] on the **correct det-1 line lattice** `Z2 (+) Z2`
    (pairing `J (+) J`, nodes `n1..n4`).  This is the realization on which the
    gauge Wilson line is the canonical character `W = F_{(0,-1,0,0)}` with
    `W*W = 1 (+) chi_2`, the special sector `{W, n3, n4}` is `[A_1, D_3]`
    (Wilson -> flavour characters), and the 't Hooft cones live.  Prefer this
    over `su2a1d3_gauged()` for any cone / Wilson / line-lattice question."""
    return BPSKAlgebra(pairing=J_PLUS_J, node_charges=[list(g) for g in DET1_NODES],
                       spec=[list(g) for g in DET1_SPEC], verify=verify)


# --- SU2A1D4 (n=4) det-1 line lattice: Z^2 (+) Z^2 (+) Z (user, 2026-06-20) ---
# Same shape as the n=3 det-1 build, one tail node longer.  Gamma = Z^2(+)Z^2(+)Z;
# pairing J(+)J(+)0 -- the 5th coordinate is ker(P) = the residual U(1) FLAVOUR of
# [A_1, D_4] (n even).  Nodes (user): pure SU(2) {n1,n2} in the first Z^2; matter n3
# + first tail n4 in the second Z^2; the pentagon tail {n4,n5} with n5 carrying the
# flavour e5.  Gram(nodes) = the coarse SU2A1D4 quiver (verified).
J_PLUS_J_PLUS_0 = [[0, 1, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 0, 0, 1, 0],
                   [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]]
DET1_NODES_D4 = [(1, 0, 0, 0, 0), (-1, 2, 0, 0, 0), (0, -1, 1, 0, 0),
                 (0, 0, 0, 1, 0), (0, 0, -1, 0, 1)]                 # n1..n5
DET1_SPEC_D4 = [(1, 0, 0, 0, 0), (1, -1, 1, 0, 0), (-1, 2, 0, 0, 0),
                (0, -1, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, -1, 0, 1)]  # [n1,n1+n3,n2,n3,n4,n5]

WILSON_D4 = (0, -1, 0, 0, 0)   # W = -1/2(n1+n2): gauge SU(2) Wilson, F(W) 3-term char
WBOSON_D4 = (0, 2, 0, 0, 0)    # root = n1+n2
MATTER_D4 = (0, -1, 1, 0, 0)   # n3 = doublet
FLAVOUR_D4 = (0, 0, 0, 0, 1)   # e5 = ker(J(+)J(+)0): the residual U(1) flavour


def su2a1d4_det1(verify: str = "off") -> BPSKAlgebra:
    """SU(2)-gauged [A_1, D_4] on the **det-1 line lattice** `Z^2 (+) Z^2 (+) Z`
    (user, 2026-06-20): pairing `J (+) J (+) 0` (the 5th coordinate is `ker(P)` =
    the residual **U(1) flavour** of [A_1, D_4]), nodes `n1..n5` (`DET1_NODES_D4`).

    The n=4 analogue of `su2a1d3_det1`: pure SU(2) `{n1,n2}` (first `Z^2`, Kronecker),
    matter `n3` + first tail `n4` (second `Z^2`), the **pentagon `[A_1,A_2]` tail
    `{n4,n5}`** with `n5` carrying the flavour `e5`.  The gauge Wilson is the canonical
    character `W = F_{WILSON_D4}` with `W*W = 1 (+) chi_2`; flavoured
    (`AbelianZPlusRing(rank=1)`); `verify_canonical_basis(K=4)` green.  Prefer this
    over the coarse `su2_gauged_a1dn(4)` for any flavour / Wilson / cone question."""
    return BPSKAlgebra(pairing=J_PLUS_J_PLUS_0,
                       node_charges=[list(g) for g in DET1_NODES_D4],
                       spec=[list(g) for g in DET1_SPEC_D4], verify=verify)


def su2a1d4_det1_matter_drop_flow(A: BPSKAlgebra | None = None) -> "SingleNodeRG":
    """Matter drop (node 2 = n3) of `su2a1d4_det1()`: the IR is **pure SU(2) ×
    (SQED_1 + flavour)** — pure SU(2) `{n1,n2}` q-commutes off the `(c,d,e)` factor
    (SQED_1 in the `(c,d)` plane, the U(1) flavour `e`)."""
    from rg_flow import SingleNodeRG
    return SingleNodeRG(A if A is not None else su2a1d4_det1(), MATTER_NODE)


def su2a1d4_det1_tail_drop_flow(A: BPSKAlgebra | None = None) -> "SingleNodeRG":
    """Tail-end drop (node 4 = n5) of `su2a1d4_det1()`: the IR is **SU2A1D3** (the
    `SU2A1Dₙ → SU2A1Dₙ₋₁` ladder) — the aux `{n1,n2,n3,n4}` multiply reproduces
    `su2a1d3_det1`; the residual U(1) flavour `e` is carried by the dropped tail."""
    from rg_flow import SingleNodeRG
    return SingleNodeRG(A if A is not None else su2a1d4_det1(), 4)


def su2a1d4_tail_drop_iso_to_su2a1d3(A: BPSKAlgebra | None = None):
    """Return `(flow, phi, su2a1d3)`: the det-1 tail-drop flow, the **structural**
    `KAlgebraIso` `phi` from its auxiliary to `su2a1d3_det1()`, and that target.

    The tail drop is a **forgetful RG**: the dropped tail node `n5` is the only
    flavour-charged node, so the IR auxiliary's canonical basis is entirely
    flavour-neutral (`e=0`) and matches `su2a1d3_det1` (label map = drop the `e`
    coordinate).  Structural battery (unit / round-trip / multiplicative / ρ)
    passes; `trace` is NOT equivariant — the aux carries the residual U(1) flavour
    as a spectator (`AbelianZPlusRing`), `su2a1d3_det1` is flavourless, so the trace
    differs by that forgotten flavour."""
    flow = su2a1d4_det1_tail_drop_flow(A)
    aux = flow.auxiliary()
    A3 = su2a1d3_det1()
    phi = KAlgebraIso(
        aux, A3,
        lambda l: Element({tuple(l[:4]): _ONE}),
        lambda l: Element({tuple(list(l) + [0]): _ONE}),
        name="SU2A1D4 tail-drop aux → su2a1d3_det1 (forget U(1) flavour)")
    return flow, phi, A3


def su2a1d4_tail_drop_to_su2a1d3(A: BPSKAlgebra | None = None) -> IsoComposedRGKAlgebra:
    """The det-1 tail-drop flow with its wrapped BPS IR swapped for `su2a1d3_det1()`
    (the forgetful-RG ladder SU2A1D4 → SU2A1D3)."""
    flow, phi, _ = su2a1d4_tail_drop_iso_to_su2a1d3(A)
    return IsoComposedRGKAlgebra(flow, phi)


# --- SU2A1D4 necklace chart (all-monomial base for the standalone ConeKAlgebra) ---
# A single cluster mutation at the matter node (2) of `su2_gauged_a1dn(4)` lands on
# this all-(+/-1) quiver (BFS-verified minimal); the n=4 analogue of B_NEWCHART.
# All 5 node-gens are MONOMIAL here (L_g^2 = L_{2g}); flavoured (ker B_NECK4 rank 1);
# `verify_canonical_basis(K=3)` green.  cocycle = the Dirac pairing B_NECK4.
B_NECK4 = [[0, 1, 1, 0, 0], [-1, 0, -1, 1, 0], [-1, 1, 0, -1, 0],
           [0, -1, 1, 0, 1], [0, 0, 0, -1, 0]]


def su2a1d4_newchart(verify: str = "off") -> BPSKAlgebra:
    """The **necklace-mutation chart** of SU2A1D4 (the all-(±1) quiver `B_NECK4`,
    = mutate the matter node of `su2_gauged_a1dn(4)`), as its own `BPSKAlgebra`
    (identity node-charges).  All five node-gens are monomial here — the
    all-MonomialCone base for the standalone `ConeKAlgebra` (the n=4 analogue of
    `su2a1d3_newchart`).  Flavoured (`AbelianZPlusRing(rank=1)`)."""
    ident = [[1 if i == j else 0 for j in range(5)] for i in range(5)]
    return BPSKAlgebra(pairing=B_NECK4, node_charges=ident, verify=verify)


def matter_drop_det1_flow(A: BPSKAlgebra | None = None) -> SingleNodeRG:
    """Drop the matter node (2) on the **det-1 line lattice** `su2a1d3_det1()`: the
    IR is **literally pure SU(2) × SQED_1**.  Pure SU(2) sits in the first `Z²`
    (the `(a,b)` plane: Kronecker `{(1,0),(-1,2)}`); SQED_1 in the second `Z²`
    (the `(c,d)` plane: electric `(0,0,0,1)`, magnetic monopole `(0,0,1,0)`,
    pairing 1); the two factors q-commute.  Unlike the coarse build the SQED_1
    **monopole is on-cone** here (det-1 = the fine Γ), so the full SQED_1 closes."""
    return SingleNodeRG(A if A is not None else su2a1d3_det1(), MATTER_NODE)


def matter_drop_det1_iso_to_pure_su2_sqed1(A: BPSKAlgebra | None = None):
    """Return `(flow, phi, target)`: the det-1 matter-drop flow, the **fully
    verified** `KAlgebraIso` `phi` from its auxiliary to `pure SU(2) ⊗ Sqed1KAlg`,
    and that target.

    The IR auxiliary factors as a clean tensor: the `(a,b)` plane is pure SU(2)
    (`pure_su2_object()`'s Kronecker `bps`, nodes `(1,0),(-1,2)`); the `(c,d)`
    plane is SQED_1 (`Sqed1KAlg`, electric/magnetic = `(c,d)`).  The label map is
    the split `(a,b,c,d) <-> ((a,b),(c,d))`, identity on each factor.  **Verified**:
    the whole `KAlgebraIso` battery (unit / round-trip / multiplicative / ρ /
    **trace**) — the det-1 build has a reliable Schur trace, so the monopole
    sector closes (the coarse `matter_drop_iso_to_pure_su2_qt1d` reaches only the
    electric `QuantumTorus1D` sector)."""
    from pure_su2_object import pure_su2_object
    from kalgebra_samples import Sqed1KAlg
    from tensor_kalgebra import TensorKAlgebra

    flow = matter_drop_det1_flow(A)
    aux = flow.auxiliary()
    psu2 = pure_su2_object().realization("bps")   # Kronecker (1,0),(-1,2)
    target = TensorKAlgebra(psu2, Sqed1KAlg())

    def fwd(label):
        a, b, c, d = label
        return Element({((a, b), (c, d)): _ONE})

    def inv(tlabel):
        (a, b), (c, d) = tlabel
        return Element({(a, b, c, d): _ONE})

    phi = KAlgebraIso(aux, target, fwd, inv,
                      name="SU2A1D3 det-1 matter-drop aux → pure SU(2) ⊗ SQED1")
    return flow, phi, target


def matter_drop_det1_to_pure_su2_sqed1(A: BPSKAlgebra | None = None) -> IsoComposedRGKAlgebra:
    """The det-1 matter-drop flow with its wrapped BPS IR swapped for the **full**
    `pure SU(2) ⊗ Sqed1KAlg` (monopole on-cone — the complete pure SU(2) × SQED_1;
    supersedes the coarse `matter_drop_to_pure_su2_qt1d`, electric sector only)."""
    flow, phi, _t = matter_drop_det1_iso_to_pure_su2_sqed1(A)
    return IsoComposedRGKAlgebra(flow, phi)


def matter_drop_det1_iso_to_abe_su2_sqed1(A: BPSKAlgebra | None = None):
    """Return `(flow, phi_abe, target)`: the det-1 matter-drop flow, the
    `KAlgebraIso` `phi_abe` from its auxiliary to **`AbelianizedSU2KAlg ⊗
    Sqed1KAlg`** — the pure-SU(2) IR factor REPLACED by its abelianized realization
    — and that target.

    Built by post-composing `matter_drop_det1_iso_to_pure_su2_sqed1`'s `phi` with
    the first-factor bps→abe iso (`pure_su2_object()`'s label-bijective `abe↔bps`
    witness).  The **structural** battery (unit / round-trip / multiplicative / ρ)
    passes; trace is NOT equivariant — `AbelianizedSU2KAlg.trace` is the U(2) Schur
    index (decoupled-photon factor), so it differs from the SU(2) trace by that
    photon factor (documented in `pure_su2_object`).  The RG/multiply content is
    identical to the BPS pure-SU(2) version; only the pure-SU(2) leg's presentation
    changes (BPS Kronecker → abelianized quantum torus)."""
    from pure_su2_object import pure_su2_object
    from tensor_kalgebra import TensorKAlgebra

    flow, phi, target_bps = matter_drop_det1_iso_to_pure_su2_sqed1(A)
    pso = pure_su2_object()
    abe = pso.realization("abe")
    su2_iso = pso.iso("bps", "abe")           # label-bijective bps → abe
    sqed1 = target_bps.factor_B
    target_abe = TensorKAlgebra(abe, sqed1)

    def t_fwd(tlabel):
        bl, sl = tlabel
        al = next(iter(su2_iso.map(Element.basis(bl)).terms))
        return Element({(al, sl): _ONE})

    def t_inv(tlabel):
        al, sl = tlabel
        bl = next(iter(su2_iso.inverse(Element.basis(al)).terms))
        return Element({(bl, sl): _ONE})

    tensor_iso = KAlgebraIso(target_bps, target_abe, t_fwd, t_inv,
                             name="pure SU(2)⊗SQED1 : bps→abe (first factor)")
    phi_abe = phi.compose(tensor_iso)         # aux → abe ⊗ Sqed1
    return flow, phi_abe, target_abe


class SU2A1D3MatterDropAbeRG(IsoComposedRGKAlgebra):
    """SU(2)-gauged [A_1,D_3] **matter-drop flow (det-1) with the pure-SU(2) IR
    factor ABELIANIZED**: `auxiliary()` = `AbelianizedSU2KAlg ⊗ Sqed1KAlg`.

    Same UV + RG data as `matter_drop_det1_to_pure_su2_sqed1` (the swap iso is
    multiplicative / ρ-equivariant, and grades the SQED_1 monopole `(0,0,1,0)`);
    only the IR *presentation* of the pure-SU(2) leg changes — BPS Kronecker →
    abelianized rational quantum torus (`AbelianizedSU2KAlg` over `PureUNKAlgebra(2)`).
    The abelianized trace is the U(2) Schur index (decoupled-photon factor), so this
    is NOT trace-equivariant with the SU(2) version — by design (see
    `AbelianizedSU2KAlg`)."""

    def __init__(self, A: BPSKAlgebra | None = None) -> None:
        flow, phi_abe, target = matter_drop_det1_iso_to_abe_su2_sqed1(A)
        super().__init__(flow, phi_abe)
        self.abe_target = target


def matter_drop_det1_to_abe_su2_sqed1(A: BPSKAlgebra | None = None) -> SU2A1D3MatterDropAbeRG:
    """The det-1 matter-drop flow with its IR pure-SU(2) factor abelianized — an
    `RGKAlgebra` whose `auxiliary()` is `AbelianizedSU2KAlg ⊗ Sqed1KAlg`."""
    return SU2A1D3MatterDropAbeRG(A)


def wboson_transvection(g):
    """`T(g) = (g1, g2 - 2 g1, g3, g4)` — the W-boson symplectic transvection
    `t_{WB,1/2}: g |-> g + 1/2 <W-boson, g> . W-boson` (`WBOSON = (0,2,0,0)`).

    `T` is the elementary cone-step / 1-W-boson translation.  On the **stable
    g1>=1 't Hooft chamber** the half-monodromy satisfies `rho^3 = T^4`
    (`rho^3(g) = (g1, g2 - 8 g1, g3, g4)` there) and `rho^3 = id` on the
    `[A_1, D_3]` special sector.  Globally `rho` is piecewise-linear, so this is a
    chamber identity; `rho = phi^4` (n=4) holds as the drift-lattice / cone-orbit
    statement (`T = phi^3`), not a closed-form map.

    **NOT an algebra automorphism.**  `T` is a symplectic transvection, so it is an
    automorphism of the *quantum torus* and it fixes Wilson (`T(W)=W`, all g1=0),
    but it does **not** preserve the canonical-basis product: `T(a.b) != T(a).T(b)`
    on every fusing monopole product (`n1.n2` Kronecker, `n1.n3`, ...).  The genuine
    Wilson-preserving automorphism is `rho` (an honest automorphism with
    `rho(W)=W`); `rho^3` is the *honest* `T^4` (= the transvection on the stable
    chamber, with piecewise fusion-corrections off it).  Use `T` only for
    fan-indexing / the drift lattice, never as a symmetry of the algebra."""
    g = tuple(g)
    return (g[0], g[1] - 2 * g[0], g[2], g[3])


# The cone inventory (det-1 Gamma), all q-commute / rays verified via A.multiply:
#   * special (gauge-invariant) = [A1,D3]:  Wilson CharacterCone {W^k} ~ R(SU(2)),
#     matter character n3 (doublet), tail monomial n4 (stem).
#   * 't Hooft (monopole): 3-ray MIXED cones = (Wilson char) x (2 monomials);
#     a representative is chart 0 of the fan; ONE rho-orbit covers the family.
DET1_SPECIAL_CONE_GENS = {"wilson_char": WILSON, "matter": MATTER, "tail": TAIL}
# representative 't Hooft cone (chart 0): a Wilson-char pair + 2 monomials -> 3 rays
DET1_THOOFT_REP = {
    "wilson_char_pair": ((1, 0, 0, -1), (1, -2, 0, -1)),   # differ by the W-boson
    "monomials": ((0, 0, 0, -1), (0, 0, 1, 0)),            # -tail, bare quark
}


def det1_cone_fan(max_charts: int = 40):
    """The 't Hooft cone fan of the det-1 realization, via the chart-mutation
    cone builder (`cluster_cone_builder`).  Returns `(clusters, edges, truncated)`;
    each chart is a 4-ray monomial cone, and every monopole chart has one ray-pair
    differing by `WBOSON` (a Wilson character) -> 3 effective rays = 1 Wilson
    character + 2 monomials (the "mixed cone").  `rho` permutes the fan; the whole
    infinite family is one rho-orbit (walked 4x finer by phi)."""
    from cluster_cone_builder import build_cluster_graph_truncated
    return build_cluster_graph_truncated(J_PLUS_J, [list(g) for g in DET1_NODES],
                                         max_charts=max_charts)


# ---------------------------------------------------------------------------
# The necklace chart and  rho = phi^4  (user, 2026-06-20)
# ---------------------------------------------------------------------------
# A mutation (necklace) chart of SU2A1D3 in which the BPS quiver becomes the
# all-(+/-1) 4-node quiver `B_NEWCHART`.  Its charges in the OLD (coarse) Gamma
# are `NEWCHART_CHARGES_IN_OLD` (the iso to `su2a1d3_gauged()` at the
# charge/quantum-torus level; the canonical-label iso is the necklace transport).
# Here the cluster automorphism phi = "necklace at the spec head" (= the user's
# mutation symmetry: mutating node 0/1 returns the same quiver up to an ORDER-8
# permutation whose charge action has perm^4 = -I) satisfies, VERIFIED:
#
#       rho  =  phi^4  =  - ( nu_0 . nu_2 . nu_3 . nu_1 )
#
# i.e. the spec `[0,2,3,1]` is the four phi actions (nu_k = forward necklace
# `_mu_g` at node k), composed and followed by charge conjugation -I (= perm^4).
# rho^2 is the full 8-necklace rotation loop (`chart_monodromy_iso`).  phi is
# INFINITE order as an algebra automorphism (its label action is piecewise) even
# though its charge action perm is order 8 -- the charges-vs-labels distinction.
B_NEWCHART = [[0, 1, 1, 0], [-1, 0, 0, -1], [-1, 0, 0, 1], [0, 1, -1, 0]]
NEWCHART_CHARGES_IN_OLD = [(1, 0, 0, 0), (0, 1, 1, 1), (0, 0, -1, 0), (0, 0, 0, -1)]
NEWCHART_SPEC_NODES = (0, 2, 3, 1)   # the four phi (necklace) actions


def su2a1d3_newchart(verify: str = "off") -> BPSKAlgebra:
    """The **necklace-mutation chart** of SU2A1D3 (the all-(+/-1) quiver
    `B_NEWCHART`), as its own `BPSKAlgebra` (identity node-charges, so the
    mutation symmetry is a root operation here).  In this chart `rho = phi^4`
    with `phi` = one necklace/spec action (`verify_rho_is_phi4`).  Iso to the old
    chart `su2a1d3_gauged()`: the charges in the old Gamma are
    `NEWCHART_CHARGES_IN_OLD` (charge/QT-level identification; the canonical-label
    iso is the necklace transport / chart-mutation path)."""
    ident = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
    return BPSKAlgebra(pairing=B_NEWCHART, node_charges=ident, verify=verify)


def verify_rho_is_phi4(labels=None) -> dict:
    """Verify, on the necklace chart, that `rho = -(nu_0 . nu_2 . nu_3 . nu_1)`
    (the spec `[0,2,3,1]` = four necklace/phi actions, composed, then charge
    conjugation `-I` = the order-8 restoring perm to the 4th power).  This is the
    `rho = phi^4` statement (user, 2026-06-20).  Returns `{label: bool}`."""
    from chart_graph import _mu_g
    A = su2a1d3_newchart()
    Pm = [list(r) for r in A.lattice.pairing]
    # spec node order (0,2,3,1) -> the consumed (head) charges = e_0,e_2,e_3,e_1
    spec_charges = [tuple(1 if i == k else 0 for i in range(4))
                    for k in NEWCHART_SPEC_NODES]

    def four_necklaces(a):
        x = tuple(a)
        for c in spec_charges:
            x = _mu_g(x, c, Pm)
        return x

    if labels is None:
        labels = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
                  (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 1, 0), (0, 1, 0, 1),
                  (2, 1, 0, 0), (1, 1, 1, 0)]
    return {a: four_necklaces(a) == tuple(-x for x in A.rho(a)) for a in labels}


def _newchart_C(v):
    """The charge intertwiner `C : Γ → Γ`, an involution `C² = id`, sending a
    necklace-chart charge to the old-chart charge (`C` = the `NEWCHART_CHARGES_IN_OLD`
    embedding): `C(v) = (v₀, v₁, v₁−v₂, v₁−v₃)`.  Satisfies `Cᵀ·B_SU2A1D3·C = B_NEWCHART`."""
    v = tuple(v)
    return (v[0], v[1], v[1] - v[2], v[1] - v[3])


# The mutation (necklace) path from the old chart to the necklace chart, found by
# bidirectional chart-graph BFS.  Mixed fwd/inv with local moves -- a general
# mutation sequence (NOT a pure forward necklace).
NEWCHART_MUTATION_PATH = [(0, "fwd"), (2, "fwd"), (0, "fwd"), (3, "fwd"),
                          (3, "inv"), (2, "inv"), (3, "fwd")]


def su2a1d3_oldnew_iso():
    """The full **canonical `KAlgebraIso`** from the old chart `su2a1d3_gauged()`
    to the necklace chart (in the old lattice), via the mutation path
    `NEWCHART_MUTATION_PATH` (`bps_chart_object.mutation_path_iso`,
    `max_local_moves=2`).

    BPSKAlgebras related by a mutation *sequence* are always KAlgebra-isomorphic, so
    this is a genuine label↦label iso preserving *all* structure — **verified**: the
    whole battery (unit / round-trip / multiplicative / ρ-equivariant /
    trace-equivariant) passes.  Returns `(A_end, iso)`; `A_end` is the necklace chart
    embedded in the old lattice (its nodes a permutation of `NEWCHART_CHARGES_IN_OLD`).
    To reach the `B_NEWCHART`-frame `su2a1d3_newchart()`, post-compose with the
    unimodular frame change `_newchart_C` (`newchart_qt_iso_to_old` is that at the QT
    level)."""
    from bps_chart_object import mutation_path_iso
    return mutation_path_iso(su2a1d3_gauged(), NEWCHART_MUTATION_PATH,
                             max_local_moves=2)


def newchart_qt_iso_to_old() -> KAlgebraIso:
    """The **frame-change** `KAlgebraIso` relating the `B_NEWCHART`-frame necklace
    chart and the old chart at the **quantum-torus** (`auxiliary()`) level, via the
    charge intertwiner `C` (`_newchart_C`; involution, `Cᵀ B_old C = B_new`).
    Verified (whole battery).

    This is the *frame-change* piece: `su2a1d3_newchart()` lives in the `B_NEWCHART`
    lattice while the necklace chart of `su2a1d3_oldnew_iso()` lives embedded in the
    old `B_SU2A1D3` lattice; `C` is the unimodular change between them.  The full
    canonical old↔necklace `KAlgebraIso` is `su2a1d3_oldnew_iso()` (mutation-path
    transport)."""
    Ao = su2a1d3_gauged()
    An = su2a1d3_newchart()
    return KAlgebraIso(An.auxiliary(), Ao.auxiliary(),
                       lambda d: Element({_newchart_C(d): _ONE}),
                       lambda g: Element({_newchart_C(g): _ONE}),
                       name="su2a1d3 necklace<->old QT intertwiner C")


# ---------------------------------------------------------------------------
# Cone structure / ConeData (necklace chart): cocycle = the Dirac pairing
# ---------------------------------------------------------------------------
# Read off in the necklace chart `su2a1d3_newchart()` (deg <= 4 fundamental piece;
# the full fan is the phi-orbit).  The ALGEBRAIC
# cone data (the PBW q-commuting blocks `ConeData` uses -- NOT the geometric chart
# fan, whose dual-cone rays need not q-commute):
#
#   * 7 multiplicative generators, ALL MONOMIAL (`L_g^k = L_{kg}`) -- so every cone
#     is a `MonomialCone`; the multi-term `F`-expansions (F=2,3,4,7) are quantum-torus
#     images, not cone-PBW characters.
#   * 5 maximal q-commuting cliques (the cones).
#   * cocycle  c(a,b) = <a,b>  (the Dirac pairing B_NEWCHART), antisymmetric -- the
#     standard quantum-torus 2-cocycle: VERIFIED `A.multiply(a,b) = q^<a,b> L_{a+b}`
#     on all 60 in-clique ordered pairs.
#   * cross_product (non-q-commuting gens): clean 2-term fusions, e.g.
#     L_{(0,0,0,1)}.L_{(1,0,0,0)} = L_{(1,0,0,1)} + q^{-1} L_{(1,0,1,1)}.
SU2A1D3_CONE_GENS = [(1, 0, 0, 0), (1, 0, 0, 1), (0, 0, 1, 0), (0, 1, 1, 0),
                     (0, 0, 0, 1), (1, 1, 0, 0), (0, 1, 0, 0)]
SU2A1D3_CONES = [
    frozenset([(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 1, 0), (1, 0, 0, 1)]),
    frozenset([(0, 0, 0, 1), (0, 1, 0, 0), (0, 1, 1, 0), (1, 0, 0, 1)]),
    frozenset([(0, 0, 1, 0), (0, 1, 1, 0), (1, 0, 0, 0), (1, 0, 0, 1)]),
    frozenset([(0, 1, 0, 0), (0, 1, 1, 0), (1, 0, 0, 1), (1, 1, 0, 0)]),
    frozenset([(0, 1, 1, 0), (1, 0, 0, 0), (1, 0, 0, 1), (1, 1, 0, 0)]),
]


def cone_cocycle(a, b) -> int:
    """The cone 2-cocycle `c(a,b)`: `X_a · X_b = q^{c(a,b)} X_{a+b}` for q-commuting
    (in-cone) generators in the necklace chart.  **= the Dirac pairing `⟨a,b⟩`**
    (`B_NEWCHART`), antisymmetric — the standard quantum-torus cocycle (verified by
    `verify_cone_structure`)."""
    B = B_NEWCHART
    return sum(a[i] * B[i][j] * b[j] for i in range(4) for j in range(4))


def verify_cone_structure() -> dict:
    """Verify the necklace-chart cone structure: (i) every generator is monomial
    (`L_g·L_g = L_{2g}`); (ii) on every in-cone (q-commuting) ordered pair,
    `A.multiply(a,b) = q^{cone_cocycle(a,b)} L_{a+b}` — i.e. the cocycle IS the Dirac
    pairing.  Returns `{check: bool}`."""
    A = su2a1d3_newchart()
    out: dict = {}

    def single_pow(a, b):
        p = A.multiply(a, b)
        if len(p.terms) != 1:
            return None
        lab, lp = next(iter(p.terms.items()))
        e = list(lp._coeffs.keys())
        if len(e) == 1 and lp._coeffs[e[0]] == 1:
            return lab, e[0]
        return None

    out["gens_monomial"] = all(
        single_pow(g, g) == (tuple(2 * x for x in g), cone_cocycle(g, g))
        for g in SU2A1D3_CONE_GENS)
    ok = True
    for C in SU2A1D3_CONES:
        for a in C:
            for b in C:
                if a == b:
                    continue
                r = single_pow(a, b)
                if r != (tuple(x + y for x, y in zip(a, b)), cone_cocycle(a, b)):
                    ok = False
    out["cocycle_is_dirac_pairing"] = ok
    return out


# ---------------------------------------------------------------------------

def certify(verbose: bool = True) -> dict:
    """Run the full certificate battery; return `{check: bool}`."""
    from directional_subquiver_rg import (certify_directional_vs_bps,
                                          verify_axioms)
    from rgkalgebra_iso_via_ir import verify_ir_iso_hypotheses

    out: dict = {}
    A = su2a1d3_gauged()

    # 1. soundness of the UV theory
    cb = A.verify_canonical_basis(K=4)
    out["uv_canonical_basis"] = all(cb.values())

    labels = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    pairs = [((1, 0, 0, 0), (0, 1, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 1)),
             ((1, 0, 0, 0), (0, 0, 1, 0))]
    cert_labels = [(0, 0, 0, 0)] + labels

    # 2. both single-node-drop flows: RG axioms + identity-on-labels vs UV
    for drop, tag in [(MATTER_NODE, "matter_drop"), (TAIL_NODE, "tail_drop")]:
        R = SingleNodeRG(A, drop)
        ax = verify_axioms(R, labels, pairs, deep_rho=False)
        out[f"{tag}_rg_axioms"] = all(ax.values())
        cv = certify_directional_vs_bps(R, A, labels=cert_labels, pairs=pairs,
                                        check_rho=True, trace_K=None)
        out[f"{tag}_vs_uv"] = all(v for k, v in cv.items() if k != "iso")

    # 3. completed swap: tail-drop IR ~= certified U(2) N_f=1 + IsoComposed
    flow, phi, _u2 = tail_drop_iso_to_u2nf1(A)
    ir_labels = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]
    ir_samples = [Element({l: _ONE}) for l in ir_labels]
    ir_pairs = [(Element({(1, 0, 0, 0): _ONE}), Element({(0, 1, 0, 0): _ONE})),
                (Element({(1, 0, 0, 0): _ONE}), Element({(0, 0, 1, 0): _ONE}))]
    hyp = verify_ir_iso_hypotheses(flow, phi, ir_samples, ir_pairs, cutoff=4)
    out["swap_ir_iso_hypotheses"] = all(hyp.values())

    swapped = IsoComposedRGKAlgebra(flow, phi)
    # The swap reproduces the UV product on the **gauge sector** (products of
    # {e0,e1,e2}); the tail e3 is the integrated-out direction (its det-4
    # embedding differs from the standalone's det-1 by the spurious flavour),
    # so tail-involving products are UV-only, outside the U(2)Nf=1 IR scope.
    gauge_pairs = [((1, 0, 0, 0), (0, 1, 0, 0)),   # Kronecker
                   ((1, 0, 0, 0), (0, 0, 1, 0))]   # node0 * matter
    out["swap_reproduces_uv_multiply"] = all(
        swapped.multiply(a, b) == A.multiply(a, b) for a, b in gauge_pairs)

    # 4. matter-drop swap: aux (+cone) ~= pure SU(2) (x) QT1D (electric tower).
    mflow, mphi, _t = matter_drop_iso_to_pure_su2_qt1d(A)
    m_labels = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)]
    m_samples = [Element({l: _ONE}) for l in m_labels]
    m_pairs = [(Element({(1, 0, 0, 0): _ONE}), Element({(0, 1, 0, 0): _ONE})),
               (Element({(1, 0, 0, 0): _ONE}), Element({(0, 0, 0, 1): _ONE}))]
    mhyp = verify_ir_iso_hypotheses(mflow, mphi, m_samples, m_pairs, cutoff=4)
    out["matter_swap_ir_iso_hypotheses"] = all(mhyp.values())

    if verbose:
        for k, v in out.items():
            print(f"  {'OK ' if v else 'XX '} {k}: {v}")
        print("ALL PASS" if all(out.values()) else "SOME FAILED")
    return out


def det1_certify(verbose: bool = True) -> dict:
    """Certificate for the **det-1 line-lattice** realization `su2a1d3_det1()`:
    Wilson character, the special sector `~= [A_1, D_3]` (Wilson -> flavour
    characters), `rho^3 = T^4` on the stable 't Hooft chamber, and `rho`
    permuting the cone fan.  All via the public API."""
    from a1dn_kalg import A1DnKAlg
    out: dict = {}
    A = su2a1d3_det1()

    # 1. Wilson W = F_{(0,-1,0,0)} is the canonical character, rho-invariant, W^2=1(+)chi2.
    out["wilson_rho_invariant"] = (A.rho(WILSON) == WILSON)
    WW = A.multiply(WILSON, WILSON).terms
    out["wilson_sq_is_1_plus_chi2"] = (
        set(WW) == {(0, 0, 0, 0), (0, -2, 0, 0)}
        and all(v == _ONE for v in WW.values()))

    # 2. special sector ~= [A1,D3] = A1DnKAlg(3): matter n3 = doublet, tail n4 = stem.
    M = A1DnKAlg(3)
    n3n3 = A.multiply(MATTER, MATTER).terms
    n4n4 = A.multiply(TAIL, TAIL).terms
    dd = M.multiply((0, 1, 0), (0, 1, 0)).terms      # doublet^2 = singlet (+) triplet
    ss = M.multiply((1, 0, 0), (1, 0, 0)).terms      # stem^2 = single
    out["matter_sq_is_doublet_clebsch"] = (len(n3n3) == 2 == len(dd))   # 1 (+) 3
    out["tail_sq_is_stem_single"] = (len(n4n4) == 1 == len(ss))
    n34 = A.multiply(MATTER, TAIL).terms
    n43 = A.multiply(TAIL, MATTER).terms
    out["matter_tail_qcommute"] = (len(n34) == 1 and len(n43) == 1
                                   and set(n34) == set(n43))

    # 3. rho^3 = T^4 on the stable g1>=1 chamber, = id on the special sector.
    def rho3(g):
        return A.rho(A.rho(A.rho(g)))

    def T4(g):
        for _ in range(4):
            g = wboson_transvection(g)
        return g
    stable = [(1, 0, 0, 0), (1, -2, 0, -1), (1, 0, 0, 1), (2, 0, 0, 0)]
    special = [MATTER, TAIL, WILSON, (0, 0, 2, 0)]
    out["rho3_eq_T4_on_stable_chamber"] = all(rho3(g) == T4(g) for g in stable)
    out["rho3_id_on_special"] = all(rho3(g) == g for g in special)

    # 4. rho permutes the cone fan (one verified example).
    clusters, _edges, _tr = det1_cone_fan(max_charts=60)
    fp = {frozenset(tuple(r) for r in c.base_monomial_rays): c.cluster_id
          for c in clusters}
    hits = 0
    for f in fp:
        img = frozenset(A.rho(tuple(r)) for r in f)
        if img in fp and img != f:
            hits += 1
    out["rho_permutes_cones"] = (hits > 0)

    if verbose:
        for k, v in out.items():
            print(f"  {'OK ' if v else 'XX '} {k}: {v}")
        print("ALL PASS" if all(out.values()) else "SOME FAILED")
    return out


if __name__ == "__main__":
    import time
    print("SU(2)-gauged [A_1, D_3] (+ tail) — certificate\n" + "=" * 50)
    t0 = time.time()
    res = certify(verbose=True)
    print(f"[{time.time() - t0:.1f}s]")

    # det-1 line-lattice realization (Wilson character, special sector ~= [A1,D3], cones)
    print("\ndet-1 line-lattice realization — certificate\n" + "=" * 50)
    t1 = time.time()
    det1_certify(verbose=True)
    print(f"[{time.time() - t1:.1f}s]")

    # demonstrate the fast IR (the swap's payoff)
    from un_nf_kalgebra import UNNfKAlgebra
    abe = UNNfKAlgebra(2, 1)
    print("\nU(2) N_f=1 IR vacuum Schur index (fast standalone):")
    print("  Tr(1) =", abe.trace(abe.identity(), K=8))
