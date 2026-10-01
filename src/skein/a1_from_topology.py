"""End-to-end lookup:  class-S A_1 genus-0 BPS quiver + finite S from
a target topology  (n_punctures, [2k_1, ..., 2k_m]) .

Migrated from `the archived tree`  on 2026-05-15 per the design record
(A_1 migration).  Rename + minimal-adaptation pass; semantics
unchanged.  Internal cross-reference  ``surface_triangulation``  is
rewired to  ``class_s.a1_surface`` ; all other imports
( ``bps_quiver_dictionary`` ,  ``bps_quiver_tools`` ,  ``pure_ade`` )
point at the canonical root-level surface.

Implements the sausage-chain + cuts pipeline described by the author:

  1. Choose a starting  SU2_chain_flavoured(N, left_nf, right_nf)
     whose initial surface-topology is ancestor-compatible with the
     target (right puncture count, right number of boundaries, all
     boundary marks initially at 2).
  2. Apply a sequence of  cut_edge  operations:
       - p-p cuts between consecutive chain punctures, creating new
         2-mark boundaries (target has m boundaries, start has m_0 ;
         need  m - m_0  p-p cuts).
       - p-b cuts of gauge arcs, each growing an existing boundary's
         mark count by 2.
  3. Compute the BPS quiver from the final triangulation via FST;
     derive a spec via  BPSQuiver.find_negating_sequence .

Endpoint configurations (the author's clarification):

  - (Nf=2, Nf=2)  :  sphere with punctures only, 0 boundaries
  - (Nf=2, Nf=1)  :  sphere + 1 boundary with 2 marks
  - (Nf=1, Nf=1)  :  sphere + 2 boundaries with 2 marks each

The unified  N -formula:  N = n + sum(k_i) + m - 3 .
"""

from __future__ import annotations

from typing import Sequence

from bps_quiver_dictionary import QuiverEntry, freeze, signature
from bps_quiver_tools import BPSQuiver
from a1_surface import Triangulation


# ---------------------------------------------------------------------
# Paired driver
# ---------------------------------------------------------------------

def cut_and_freeze(tri: Triangulation,
                    entry: QuiverEntry | None,
                    edge_index: int) -> tuple[Triangulation, QuiverEntry | None]:
    """Simultaneously cut the triangulation along  edge_index  and
    freeze the corresponding quiver node.

    If  entry  is  None , only the triangulation is updated (the
    caller will derive the final quiver via FST at the end).  If
    provided, the entry's exchange matrix + spec are updated via
    the standard  bps_quiver_dictionary.freeze  operation and the
    new sub-entry is returned.
    """
    new_entry = None
    if entry is not None:
        node_pos = tri.internal_edge_indices.index(edge_index)
        new_entry = freeze(entry, node_pos)
    new_tri = tri.cut_edge(edge_index)
    return new_tri, new_entry


# ---------------------------------------------------------------------
# Edge finders (mechanical helpers for the planner)
# ---------------------------------------------------------------------

def _find_pp_edge_between_punctures(tri: Triangulation) -> int | None:
    """Return the index of an internal edge whose endpoints are two
    DIFFERENT punctures (non-self-loop, p-p).  Returns None if none
    exists."""
    for i, e in enumerate(tri.edges):
        if not e.internal or e.is_self_loop:
            continue
        if (tri.vertices[e.v1].kind == "puncture"
                and tri.vertices[e.v2].kind == "puncture"):
            return i
    return None


def _find_pb_edge_for_boundary(tri: Triangulation,
                                bdry: int) -> int | None:
    """Return the index of an internal edge that is puncture-to-
    boundary-mark, where the marked endpoint is on boundary ``bdry``.
    Returns None if none exists."""
    for i, e in enumerate(tri.edges):
        if not e.internal or e.is_self_loop:
            continue
        k1 = tri.vertices[e.v1].kind
        k2 = tri.vertices[e.v2].kind
        if k1 == "puncture" and k2 == "marked":
            if tri.vertices[e.v2].boundary == bdry:
                return i
        elif k2 == "puncture" and k1 == "marked":
            if tri.vertices[e.v1].boundary == bdry:
                return i
    return None


# ---------------------------------------------------------------------
# Planner  +  user function
# ---------------------------------------------------------------------

def _tropical_mutate_charge(charge: tuple[int, ...],
                              B: Sequence[Sequence[int]],
                              k: int) -> tuple[int, ...]:
    """Tropical mutation of a lattice charge at node  k  under pairing  B .

    Formula (for standard-basis quiver before mutation):
      new[j] = charge[j]                                 for j != k
      new[k] = -charge[k] + sum_{i != k} max(0, B[i][k]) * charge[i]

    This expresses the original-basis charge in the post-mutation
    std basis (Q_new's "e_j" = the new node charges).
    """
    n = len(charge)
    new = list(charge)
    new[k] = -charge[k]
    for i in range(n):
        if i == k:
            continue
        new[k] += max(0, B[i][k]) * charge[i]
    return tuple(new)


def _tropical_reverse_mutate_charge(charge: tuple[int, ...],
                                      B: Sequence[Sequence[int]],
                                      k: int) -> tuple[int, ...]:
    """Tropical **inverse** mutation of a lattice charge at node  k
    under pairing  B .

    Derived from  :func:`_tropical_mutate_charge`  by swapping
    ``max(0, B[i][k])``  for  ``max(-B[i][k], 0)``  (the sign-flip
    inside  max()  that distinguishes reverse from forward tropical
    mutation, cf.  ``BPSQuiver.reverse_mutate``  and the paper
    eq 491).

    Used to translate spec charges to the post-inverse-mutation std
    basis in  :func:`surface_catalog.paired_inverse_necklace_step` .
    """
    n = len(charge)
    new = list(charge)
    new[k] = -charge[k]
    for i in range(n):
        if i == k:
            continue
        new[k] += max(-B[i][k], 0) * charge[i]
    return tuple(new)


def _mutate_matter_block(Q: "BPSQuiver",
                          spec: Sequence[tuple],
                          node_k: int,
                          block_size: int) -> tuple["BPSQuiver", tuple[tuple[int, ...], ...]] | None:
    """Transport spec through a single quiver mutation at matter node
    ``node_k`` , treating the corresponding matter block as an
    operator-commuting unit.

    Algorithm (the author's recipe):
      1. Locate the  ``block_size``  consecutive spec factors
         starting at the simple factor  ``e_{node_k}`` .
      2. Move that block to the HEAD of the spec (commutes with the
         remaining factors at the block-operator level; verified by
         ``verify_spectrum_generator``  on the reordered spec).
      3. Mutate the quiver at  ``node_k`` .  Remove the head factor
         and append  ``-moved``  at the tail of the (remaining) spec.
      4. Verify the new spec is a negating sequence on the mutated
         quiver.

    Returns  ``(Q_new_std, new_spec_in_new_std_basis)``  on success,
    or  None  if any verification step fails.  Q_new_std  is a fresh
    std-basis quiver with the mutated exchange matrix; the returned
    spec has been tropically translated so each factor is in that
    new std basis (verify-ready).
    """
    n = Q.n_nodes
    e_k = tuple(1 if j == node_k else 0 for j in range(n))
    start = None
    for i, g in enumerate(spec):
        if tuple(g) == e_k:
            start = i
            break
    if start is None:
        return None
    block = list(spec[start:start + block_size])
    if len(block) < block_size:
        return None
    rest = list(spec[:start]) + list(spec[start + block_size:])
    reordered = block + rest
    ok, _ = Q.verify_spectrum_generator(reordered)
    if not ok:
        return None
    # Capture B for tropical mutation of spec factors.
    B_before = [list(r) for r in Q.exchange]
    Q_new = Q.mutate(node_k)
    moved = reordered[0]
    neg = tuple(-x for x in moved)
    new_spec_old_basis = tuple(reordered[1:]) + (neg,)
    # Translate each factor from pre-mutation std basis to post-mutation
    # std basis via tropical mutation at  node_k .
    new_spec = tuple(
        _tropical_mutate_charge(tuple(g), B_before, node_k)
        for g in new_spec_old_basis
    )
    # Build a fresh std-basis quiver with the mutated exchange; verify.
    from bps_quiver_tools import BPSQuiver as _BPSQuiver
    std = [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]
    Q_new_std = _BPSQuiver(
        charges=std, frozen=[False] * n,
        exchange_matrix=[list(r) for r in Q_new.exchange],
    )
    ok2, _ = Q_new_std.verify_spectrum_generator(list(new_spec))
    if not ok2:
        return None
    return Q_new_std, new_spec


def _mutate_at_node(spec: Sequence[tuple],
                     Q_before: "BPSQuiver",
                     node_idx: int) -> tuple[tuple[tuple[int, ...], ...], "BPSQuiver"] | None:
    """Mutate  Q_before  at  node_idx  and transport the spec.

    The author's recipe: the factor whose charge equals  Q_before.charges[
    node_idx]  corresponds to a matter  E_q  that can be moved safely
    to the head of  S .  After mutation at that node, the new spec is
    ``rest + [-moved]``  where  moved  is the head factor and  rest
    is the spec minus  moved .

    Returns ``(new_spec, Q_after)``  or  None  if
    ``Q_before.charges[node_idx]``  is not a single factor in
    ``spec``  (e.g. the corresponding quiver node's charge is not a
    simple generator at this step).  The returned spec should be
    verified by  ``Q_after.verify_spectrum_generator``  at the call
    site.
    """
    moved = tuple(Q_before.charges[node_idx])
    rest: list[tuple] = []
    found = False
    for g in spec:
        if (not found) and tuple(g) == moved:
            found = True
            continue
        rest.append(tuple(g))
    if not found:
        return None
    Q_after = Q_before.mutate(node_idx)
    neg_moved = tuple(-x for x in moved)
    return tuple(rest + [neg_moved]), Q_after


def _find_node_permutation(B_tri: Sequence[Sequence[int]],
                            B_luq: Sequence[Sequence[int]]) -> tuple[int, ...] | None:
    """Brute-force permutation  sigma  on  range(n)  such that
    ``B_tri[j][k] == B_luq[sigma(j)][sigma(k)]``  for all  j, k .

    Used to transplant  LinearUQuiver 's spec onto the triangulation's
    FST node ordering.  O(n!) ; tolerable for  n <= 10  in practice.
    Returns None if no such permutation exists.
    """
    import itertools
    n = len(B_tri)
    if len(B_luq) != n:
        return None
    for sigma in itertools.permutations(range(n)):
        ok = all(
            B_tri[j][k] == B_luq[sigma[j]][sigma[k]]
            for j in range(n) for k in range(n)
        )
        if ok:
            return sigma
    return None


def _choose_endpoints(m_target: int, n_odd_target: int) -> tuple[int, int]:
    """Pick  (left_nf, right_nf)  so the starting surface has the
    correct number of starting boundaries AND enough odd-mark
    starting boundaries to absorb the odd-mark targets.

    Endpoint mark counts:
      - Nf=0 : 1 mark (ODD)
      - Nf=1 : 2 marks (EVEN)
      - Nf=2 : no boundary

    Constraint:  n_odd_target <= 2  (only the 2 endpoint slots can
    start a boundary with an odd number of marks; p-p cut
    boundaries always start at 2 marks = even).
    """
    if n_odd_target > 2:
        raise ValueError(
            f"cannot produce more than 2 odd-mark boundaries (only the "
            f"two Nf=0 endpoint slots give odd starts); got "
            f"n_odd={n_odd_target}"
        )
    if m_target == 0:
        return (2, 2)
    if m_target == 1:
        return (0, 2) if n_odd_target == 1 else (2, 1)
    # m_target >= 2.
    if n_odd_target == 2:
        return (0, 0)
    if n_odd_target == 1:
        return (0, 1)
    return (1, 1)


def _find_mm_earcut_on_boundary(tri: Triangulation,
                                 bdry: int) -> int | None:
    """Find an internal edge suitable for an (a, a+2)-same-boundary
    m-m cut: both endpoints marked on boundary  ``bdry`` , and the
    adjacent triangle has all 3 edges boundary-on-``bdry``  (so
    cutting disconnects the triangle)."""
    for i, e in enumerate(tri.edges):
        if not e.internal or e.is_self_loop:
            continue
        if (tri.vertices[e.v1].kind != "marked"
                or tri.vertices[e.v2].kind != "marked"):
            continue
        if (tri.vertices[e.v1].boundary != bdry
                or tri.vertices[e.v2].boundary != bdry):
            continue
        # Check one of the adjacent triangles is fully on  bdry .
        adj = [ti for ti, T in enumerate(tri.triangles) if i in T.edges]
        if len(adj) != 2:
            continue
        for ti in adj:
            T = tri.triangles[ti]
            others = [eidx for eidx in T.edges if eidx != i]
            if all(
                (not tri.edges[oi].internal)
                and tri.edges[oi].v1 != tri.edges[oi].v2
                and tri.vertices[tri.edges[oi].v1].boundary == bdry
                and tri.vertices[tri.edges[oi].v2].boundary == bdry
                for oi in others
            ):
                return i
    return None


def class_S_A1_bps(
    n_punctures: int,
    marks_list: Sequence[int] = (),
    genus: int = 0,
) -> "BPSKAlgebra":
    """Build the BPS K-algebra of the class-S A_1 theory with the
    given topology.

    Parameters
    ----------
    n_punctures:
        Number of regular (Nf=2) punctures.
    marks_list:
        Mark counts for each marked boundary component (genus=0 only).
    genus:
        Surface genus.  ``0`` = sphere (default); ``1`` = torus.

    **Genus 0 (sphere).**  Only the ``(Nf=2, Nf=2)``  no-cuts case is
    wired in: ``n_punctures >= 5`` ,  ``marks_list = ()`` ,  ``N = n - 3``
    SU(2) gauge factors, rank ``3N + 3`` .  Uses the improved-FST
    presentation with the ``(1,1)/(1,-1)``  diamond flavour convention
    at each cap (see :func:`_build_sw_chamber_n_sphere`).

    **Genus 1 (torus).**  Circular sausage chain with no matter:
    ``n_punctures >= 2`` ,  ``marks_list = ()`` ,  ``N = n``  SU(2) gauge
    factors in a closed ring, ``N``  bifund nodes, rank ``3N`` .  Each
    puncture corresponds to one bifundamental neck; there are no
    flavour/matter nodes (see :func:`_build_sw_chamber_n_torus`).
    """
    from bps_kalgebra import BPSKAlgebra as _BPSKAlgebra

    if genus == 0:
        if list(marks_list) != []:
            raise NotImplementedError(
                f"class_S_A1_bps genus=0: marks_list != () not yet supported "
                f"in the BPSKAlgebra surface.  Use "
                f"legacy.class_s_a1_bps.class_S_A1_bps(...) for the legacy "
                f"dict surface."
            )
        N = n_punctures - 3
        if N < 2:
            raise ValueError(
                f"sphere case requires n_punctures >= 5 (N >= 2); "
                f"got n_punctures = {n_punctures}."
            )
        sw = _build_sw_chamber_n_sphere(N)
        return _BPSKAlgebra(
            pairing=sw["pairing"],
            node_charges=sw["nodes"],
            spec=sw["spec"],
            cone_witness=_sw_chamber_n_sphere_cone_witness(N, sw),
            verify="off",
        )

    if genus == 1:
        if list(marks_list) != []:
            raise NotImplementedError(
                "class_S_A1_bps genus=1: marks_list != () not yet supported."
            )
        N = n_punctures
        if N < 2:
            raise ValueError(
                f"torus case requires n_punctures >= 2 (N >= 2); "
                f"got n_punctures = {n_punctures}."
            )
        sw = _build_sw_chamber_n_torus(N)
        return _BPSKAlgebra(
            pairing=sw["pairing"],
            node_charges=sw["nodes"],
            spec=sw["spec"],
            cone_witness=_sw_chamber_n_torus_cone_witness(N, sw),
            verify="off",
        )

    raise NotImplementedError(
        f"class_S_A1_bps: genus={genus} not yet supported."
    )


# ---------------------------------------------------------------------
# Ambient lift (PROTOTYPE)
# ---------------------------------------------------------------------
#
# The flat output of  class_S_A1_bps  bakes the FST exchange matrix in
# as the lattice pairing on standard-basis node charges.  For pure
# SU(2) that means pairing  [[0, 2], [-2, 0]]  with nodes  (1,0) ,
# (0,1) -- a non-unimodular sublattice (det = 2) where Nahm-index
# solutions only exist for charges with even second coordinate.  Worse:
# downstream tools  (F-solver canonicity, Schur-prefactor exponent,
# sigma orbits, palindromic tau)  all need patching to compensate.
#
# The right lattice is the unimodular symplectic ambient  Z^{2N_eff}
# with pairing  block-diag([[0,1],[-1,0]])  -- morally  H_1  of the
# spectral cover with its Poincare-dual symplectic form.  The two
# SU(2) gauge nodes embed as the composite charges  (1, 0)  and
# (-1, 2) ; their bracket is  2 , reproducing the FST exchange, but
# the ambient is unimodular so all the parity / canonicity issues
# disappear.
#
# This prototype implements that lift incrementally.  Stage 1 below
# covers the pure-SU(2) endpoint  (Nf=0/Nf=0, N=1, no cuts) ; later
# stages will extend to longer chains, cuts, and Nf=1 / Nf=2 caps
# (the last requires the matter-block mutation of class_s_a1_bps,
# with flavour charges (1,1), (1,-1) per the author's mutated-flavour
# convention).


def _mutate_matter_block_ambient(
    Q: "BPSQuiver",
    spec: Sequence[tuple[int, ...]],
    node_k: int,
    block_size: int,
) -> tuple["BPSQuiver", tuple[tuple[int, ...], ...]] | None:
    """Ambient-aware analogue of :func:`_mutate_matter_block` .

    Q  carries node charges in a unimodular symplectic ambient
    (with  Q.ambient_pairing  set; spec factors live in the same
    ambient).  Mutations are FZ tropical mutations on those ambient
    charges (preserving the ambient lattice automorphism), so unlike
    the standard-basis path no tropical re-coordinatisation of the
    spec is needed: the moved factor is replaced with its ambient
    negation directly.

    Returns ``(Q_after, new_spec)``  on success, or  None  if the
    candidate block / mutation does not produce a verified spec.
    """
    target = tuple(Q.charges[node_k])
    start = None
    for i, g in enumerate(spec):
        if tuple(g) == target:
            start = i
            break
    if start is None:
        return None
    block = list(spec[start:start + block_size])
    if len(block) < block_size:
        return None
    rest = list(spec[:start]) + list(spec[start + block_size:])
    reordered = block + rest
    ok, _ = Q.verify_spectrum_generator(reordered)
    if not ok:
        return None
    Q_after = Q.mutate(node_k)
    moved = reordered[0]
    neg = tuple(-x for x in moved)
    new_spec = tuple(reordered[1:]) + (neg,)
    ok2, _ = Q_after.verify_spectrum_generator(list(new_spec))
    if not ok2:
        return None
    return Q_after, new_spec


def _build_sw_chamber_n_sphere(N: int) -> dict:
    """SW-chamber ambient lift for the n-puncture sphere with
    ``N = n - 3``  SU(2) gauge factors and Nf=2 caps on both ends.

    Builds the pairing, node charges, and spec natively (without
    going through  SUN_bifund / LinearUQuiver) so that the gauge /
    flavour split is explicit in the slot layout and so the diamond
    flavour convention -- matter charges at  ``(1, 1)``  and
    ``(1, -1)``  in the cap's shared Z^2 -- is built in directly.

    Returns a dict with keys ``pairing``, ``nodes``, ``spec`` ;
    rank  3N + 3 = 3n - 6 .
    """
    if N < 2:
        raise ValueError(
            f"SW chamber for n-puncture sphere requires N >= 2 "
            f"(equivalently n >= 5); got N={N}."
        )

    # Slot allocation: build positions for each block.
    #   per gauge factor i: 2 gauge slots, optionally followed by 2
    #   left/right flavour slots if i is the first/last factor.
    gauge_starts: list[int] = []
    cursor = 0
    left_flav_start: int
    right_flav_start: int
    for i in range(N):
        gauge_starts.append(cursor)
        cursor += 2
        if i == 0:
            left_flav_start = cursor
            cursor += 2
        if i == N - 1:
            right_flav_start = cursor
            cursor += 2
    bifund_flav_starts = [cursor + j for j in range(N - 1)]
    cursor += N - 1
    rank = cursor
    assert rank == 3 * N + 3, (rank, N)

    def gauge_slot(i: int, k: int) -> int:
        """k-th gauge slot of factor i  (k = 0 or 1)."""
        return gauge_starts[i] + k

    def make_vec() -> list[int]:
        return [0] * rank

    # Gauge nodes: for each factor i, gamma_i = e_{slot 0} and
    #              beta_i = -e_{slot 0} + 2 e_{slot 1} .
    gauge_nodes: list[tuple[int, ...]] = []
    for i in range(N):
        v = make_vec(); v[gauge_slot(i, 0)] = 1
        gauge_nodes.append(tuple(v))
        v = make_vec(); v[gauge_slot(i, 0)] = -1; v[gauge_slot(i, 1)] = 2
        gauge_nodes.append(tuple(v))

    # Lowest weight of fund(SU(2)) at factor i: (0, -1) at gauge
    # slots of i .  Convention matches SUN_Nf.
    def lowest_weight(i: int) -> list[int]:
        v = make_vec()
        v[gauge_slot(i, 1)] = -1
        return v

    # Wilson-line support of the fundamental of SU(2) (lowest weight
    # of  F_{w_2}).  Three terms at the gauge slots of factor i :
    #   (0, -1),  (-1, 1),  (0, 1) .
    def fund_alpha_at(i: int) -> list[list[int]]:
        out: list[list[int]] = []
        v = make_vec(); v[gauge_slot(i, 1)] = -1; out.append(v)
        v = make_vec(); v[gauge_slot(i, 0)] = -1; v[gauge_slot(i, 1)] = 1; out.append(v)
        v = make_vec(); v[gauge_slot(i, 1)] = 1; out.append(v)
        return out

    # Matter nodes: m_+  =  lowest_weight(i) + (1, 1) at flavour pair
    #               m_-  =  lowest_weight(i) + (1,-1) at flavour pair
    # for the left cap (i = 0) and right cap (i = N - 1).
    def diamond(slot_lo: int, slot_hi: int, sign: int) -> list[int]:
        """(1, 1) if sign = +1, (1, -1) if sign = -1, in the cap's
        Z^2 flavour sublattice."""
        v = make_vec(); v[slot_lo] = 1; v[slot_hi] = sign
        return v

    def add(u: list[int], v: list[int]) -> list[int]:
        return [a + b for a, b in zip(u, v)]

    matter_left = [
        tuple(add(lowest_weight(0), diamond(left_flav_start, left_flav_start + 1, +1))),
        tuple(add(lowest_weight(0), diamond(left_flav_start, left_flav_start + 1, -1))),
    ]
    matter_right = [
        tuple(add(lowest_weight(N - 1), diamond(right_flav_start, right_flav_start + 1, +1))),
        tuple(add(lowest_weight(N - 1), diamond(right_flav_start, right_flav_start + 1, -1))),
    ]

    # Bifund nodes between consecutive gauge factors.  Convention from
    # SUN_bifund: bifund_e = lowest_weight(i) + lowest_weight(i+1) +
    #             e_{bifund slot of edge e} .
    bifund_nodes: list[tuple[int, ...]] = []
    for e in range(N - 1):
        v = add(lowest_weight(e), lowest_weight(e + 1))
        v[bifund_flav_starts[e]] = 1
        bifund_nodes.append(tuple(v))

    # Order the nodes:
    #   factor 1 gauge pair, left matter pair,
    #   factor 2 gauge pair, ...  (interior factors gauge only),
    #   factor N gauge pair, right matter pair,
    #   bifund nodes.
    nodes: list[tuple[int, ...]] = []
    nodes.append(gauge_nodes[0]); nodes.append(gauge_nodes[1])
    nodes.extend(matter_left)
    for i in range(1, N - 1):
        nodes.append(gauge_nodes[2 * i]); nodes.append(gauge_nodes[2 * i + 1])
    if N >= 2:
        nodes.append(gauge_nodes[2 * (N - 1)]); nodes.append(gauge_nodes[2 * (N - 1) + 1])
        nodes.extend(matter_right)
    nodes.extend(bifund_nodes)

    # Pairing: J at each gauge block, zero elsewhere.
    pairing = [[0] * rank for _ in range(rank)]
    for i in range(N):
        s0, s1 = gauge_slot(i, 0), gauge_slot(i, 1)
        pairing[s0][s1] = 1
        pairing[s1][s0] = -1

    # Spec.  Order matters -- it must be a valid negating sequence on
    # the SW-chamber quiver (we will verify).  Use the SUN_bifund /
    # LinearUQuiver convention:
    #   1. all bifund matter blocks first (in edge order),
    #   2. then per-factor matter blocks (left cap on factor 0,
    #      right cap on factor N-1) followed by the factor's gauge
    #      pair.
    # Each fundamental matter block is the 3 Wilson-line factors
    # plus the corresponding flavour direction.
    spec: list[tuple[int, ...]] = []
    # 1. Bifund blocks.
    from pure_ade import _bifund_pair_order_AN
    pair_order = _bifund_pair_order_AN(3, 3)
    for e in range(N - 1):
        alphas_i = fund_alpha_at(e)
        alphas_j = fund_alpha_at(e + 1)
        for (a, b) in pair_order:
            v = add(alphas_i[a], alphas_j[b])
            v[bifund_flav_starts[e]] = 1
            spec.append(tuple(v))
    # 2. Per-factor matter + gauge blocks.
    for i in range(N):
        if i == 0:
            for sign in (+1, -1):
                flav = diamond(left_flav_start, left_flav_start + 1, sign)
                for a in fund_alpha_at(i):
                    spec.append(tuple(add(a, flav)))
        if i == N - 1:
            for sign in (+1, -1):
                flav = diamond(right_flav_start, right_flav_start + 1, sign)
                for a in fund_alpha_at(i):
                    spec.append(tuple(add(a, flav)))
        # Gauge pair of factor i .
        spec.append(gauge_nodes[2 * i])
        spec.append(gauge_nodes[2 * i + 1])

    return {
        "pairing": pairing,
        "nodes": nodes,
        "spec": spec,
        "rank": rank,
    }


def _build_sw_chamber_n_torus(N: int) -> dict:
    """SW-chamber ambient lift for the n-punctured torus with
    ``N = n >= 2``  SU(2) gauge factors arranged in a **closed ring**
    (circular sausage chain, no Nf caps, no matter — only bifundamentals).

    Slot layout: ``2N``  gauge slots (pairs 0..2N-1) followed by
    ``N``  bifund flavour slots (2N..3N-1).  Total rank ``3N``.

    Nodes:

    * ``2N``  gauge nodes  ``γ_i = e_{2i}`` ,  ``β_i = -e_{2i} + 2 e_{2i+1}``
    * ``N``  bifund nodes  ``b_e = lw(e) + lw((e+1)%N) + e_{2N+e}``

    where  ``lw(i) = -e_{2i+1}``  (lowest weight of the fundamental
    of the i-th SU(2) factor).
    """
    if N < 2:
        raise ValueError(
            f"SW chamber for n-punctured torus requires N >= 2; got N={N}."
        )

    rank = 3 * N

    def gauge_slot(i: int, k: int) -> int:
        return 2 * i + k

    def make_vec() -> list[int]:
        return [0] * rank

    def add(u: list[int], v: list[int]) -> list[int]:
        return [a + b for a, b in zip(u, v)]

    def lowest_weight(i: int) -> list[int]:
        v = make_vec(); v[gauge_slot(i, 1)] = -1
        return v

    def fund_alpha_at(i: int) -> list[list[int]]:
        out: list[list[int]] = []
        v = make_vec(); v[gauge_slot(i, 1)] = -1; out.append(v)
        v = make_vec(); v[gauge_slot(i, 0)] = -1; v[gauge_slot(i, 1)] = 1; out.append(v)
        v = make_vec(); v[gauge_slot(i, 1)] = 1; out.append(v)
        return out

    # Gauge nodes.
    gauge_nodes: list[tuple[int, ...]] = []
    for i in range(N):
        v = make_vec(); v[gauge_slot(i, 0)] = 1
        gauge_nodes.append(tuple(v))
        v = make_vec(); v[gauge_slot(i, 0)] = -1; v[gauge_slot(i, 1)] = 2
        gauge_nodes.append(tuple(v))

    # Bifund nodes (circular: edge e connects factor e to factor (e+1) % N).
    bifund_nodes: list[tuple[int, ...]] = []
    for e in range(N):
        v = add(lowest_weight(e), lowest_weight((e + 1) % N))
        v[2 * N + e] = 1
        bifund_nodes.append(tuple(v))

    # Node order: gauge pairs first, then bifund nodes.
    nodes: list[tuple[int, ...]] = []
    for i in range(N):
        nodes.append(gauge_nodes[2 * i])
        nodes.append(gauge_nodes[2 * i + 1])
    nodes.extend(bifund_nodes)

    # Pairing: J on each gauge pair, 0 on bifund slots.
    pairing = [[0] * rank for _ in range(rank)]
    for i in range(N):
        s0, s1 = gauge_slot(i, 0), gauge_slot(i, 1)
        pairing[s0][s1] = 1
        pairing[s1][s0] = -1

    # Spec: N bifund blocks (circular order) then N gauge pairs.
    spec: list[tuple[int, ...]] = []
    from pure_ade import _bifund_pair_order_AN
    pair_order = _bifund_pair_order_AN(3, 3)
    for e in range(N):
        alphas_i = fund_alpha_at(e)
        alphas_j = fund_alpha_at((e + 1) % N)
        for (a, b) in pair_order:
            v = add(alphas_i[a], alphas_j[b])
            v[2 * N + e] = 1
            spec.append(tuple(v))
    for i in range(N):
        spec.append(gauge_nodes[2 * i])
        spec.append(gauge_nodes[2 * i + 1])

    return {
        "pairing": pairing,
        "nodes": nodes,
        "spec": spec,
        "rank": rank,
    }


def _sw_chamber_n_sphere_cone_witness(
    N: int,
    sw: dict | None = None,
) -> tuple[int, ...]:
    """Closed-form strict cone witness for the SW chamber of the
    n-puncture sphere  ``S^2_{0,n}``  with  ``N = n - 3``  SU(2) gauge
    factors and Nf=2 caps.

    Supplying this to :class:`BPSKAlgebra`  short-circuits its
    exponential box search in :func:`compute_strict_cone_witness` .

    Recipe (matches :class:`quiver_constructors.LinearUQuiver`'s
    ``cone_witness`` ):

    * Gauge witness on each factor's 2 slots: ``(1, 1)``  -- gives
      ``<f, γ_i> = 1``  and ``<f, β_i> = 1`` .
    * Cap flavour witness on each Nf=2 cap's Z^2: ``(2, 0)``  --
      gives ``<f, m_+> = -1 + 2 = 1``  and ``<f, m_-> = -1 + 2 = 1`` .
    * Per-bifund flavour slot: ``max(1 - <f, b_e>, 0)``  computed
      from the pre-bifund-slot contribution; ensures
      ``<f, b_e> ≥ 1`` .
    """
    if N < 2:
        raise ValueError(f"SW-chamber witness requires N >= 2; got {N}")
    if sw is None:
        sw = _build_sw_chamber_n_sphere(N)
    rank = sw["rank"]
    f = [0] * rank
    # Gauge: f = (1, 1) on each factor's 2 slots.
    for i in range(N):
        f[2 * i] = 1
        f[2 * i + 1] = 1
    # Cap flavour: (2, 0) on each Nf=2 cap's Z^2 (left then right).
    left_flav_start = 2 * N
    right_flav_start = 2 * N + 2
    f[left_flav_start] = 2
    f[left_flav_start + 1] = 0
    f[right_flav_start] = 2
    f[right_flav_start + 1] = 0
    # Bifund slot per edge: tighten so <f, b_e> = 1.
    bifund_start = 2 * N + 4
    bifund_node_indices = list(range(
        len(sw["nodes"]) - (N - 1), len(sw["nodes"])
    ))
    for edge in range(N - 1):
        bn = sw["nodes"][bifund_node_indices[edge]]
        val = sum(fi * bi for fi, bi in zip(f, bn))
        f[bifund_start + edge] = max(1 - val, 0)
    return tuple(f)


def _sw_chamber_n_torus_cone_witness(
    N: int,
    sw: dict | None = None,
) -> tuple[int, ...]:
    """Closed-form strict cone witness for the SW chamber of the
    n-punctured torus  (N = n >= 2)  with no matter.

    * Gauge slots: ``(1, 1)``  per factor — gives  ``<f, γ_i> = 1``
      and  ``<f, β_i> = -1 + 2 = 1`` .
    * Bifund slot per edge: ``3``  — gives
      ``<f, b_e> = -f[2e+1] - f[2(e+1)%N+1] + 3 = -1 - 1 + 3 = 1`` .
    """
    if N < 2:
        raise ValueError(f"torus cone witness requires N >= 2; got {N}")
    if sw is None:
        sw = _build_sw_chamber_n_torus(N)
    rank = sw["rank"]
    f = [0] * rank
    for i in range(N):
        f[2 * i] = 1
        f[2 * i + 1] = 1
    for e in range(N):
        f[2 * N + e] = 3
    return tuple(f)


def n_sphere_puncture_cartans(N: int) -> list[tuple[str, tuple[int, ...]]]:
    """Per-puncture flavour Cartan basis for the SW-chamber  n-puncture
    sphere lift  (n = N + 3, N >= 2) .

    Each puncture contributes one  SU(2)  flavour Cartan -- a single
    standard-basis vector of the SW-chamber ambient.  The two cap
    pairs are the  Cartan  ⊕  Cartan  decomposition of  Spin(4) =
    SU(2)_L × SU(2)_R  for each Nf=2 cap; the bifund punctures are
    Cartans of the SU(2) flavour symmetry on each real bifundamental
    (the  U(1)  enhanced to  SU(2)  by the reality of the (2, 2)
    rep), normalized so  T_3 = +/-1  are the highest / lowest weights
    of its fundamental.

    Returns a list of  ``(puncture_name, e_slot)``  tuples, of length
    n , in left-to-right puncture order along the chain:

        L_L,  L_R,  B_0,  B_1,  ...,  B_{N-2},  R_L,  R_R .

    These  n  vectors form a  Z-basis of  ker B  (index 1) -- so the
    flavour lattice is genuinely  Z^n , one Cartan per puncture, no
    parity / index-4 mismatch.
    """
    if N < 2:
        raise ValueError(f"n-puncture sphere lift requires N >= 2; got N={N}")
    cursor = 0
    rank = 3 * N + 3
    out: list[tuple[str, tuple[int, ...]]] = []

    def e(slot: int) -> tuple[int, ...]:
        v = [0] * rank
        v[slot] = 1
        return tuple(v)

    for i in range(N):
        cursor += 2  # gauge slots
        if i == 0:
            out.append(("L_L  (T_3^L  of Spin(4)_left)",  e(cursor)))
            out.append(("L_R  (T_3^R  of Spin(4)_left)",  e(cursor + 1)))
            cursor += 2
        if i == N - 1:
            out.append(("R_L  (T_3^L  of Spin(4)_right)", e(cursor)))
            out.append(("R_R  (T_3^R  of Spin(4)_right)", e(cursor + 1)))
            cursor += 2
    # Bifund punctures sit at the tail of the slot allocation.  Insert
    # them between the cap groups in puncture order  (between L pair
    # and R pair).
    bifund_cartans = []
    for e_idx in range(N - 1):
        bifund_cartans.append(
            (f"B{e_idx}  (T_3  of SU(2) on real bifund {e_idx})",
             e(cursor))
        )
        cursor += 1
    # Re-order so the result matches puncture order along the chain:
    # L_L, L_R, B_0, ..., B_{N-2}, R_L, R_R.
    if N >= 2:
        return out[:2] + bifund_cartans + out[2:]
    return out + bifund_cartans


def _verify_ambient(pairing, node_charges, spec) -> tuple[tuple[int, ...], ...]:
    """Compute  exchange[i][j] = <node_i, node_j>  under  pairing  and
    verify  spec  is a valid negating sequence on a fresh BPSQuiver
    built from  (node_charges, pairing) .  Returns the exchange matrix.
    """
    rank = len(pairing)
    n = len(node_charges)
    exchange = tuple(
        tuple(
            int(sum(
                pairing[a][b] * node_charges[i][a] * node_charges[j][b]
                for a in range(rank) for b in range(rank)
            ))
            for j in range(n)
        )
        for i in range(n)
    )
    Q = BPSQuiver.from_pairing(
        [tuple(c) for c in node_charges],
        [list(r) for r in pairing],
    )
    ok, _ = Q.verify_spectrum_generator([tuple(g) for g in spec])
    if not ok:
        raise RuntimeError(
            "ambient lift: spec failed verify_spectrum_generator on the "
            "(pairing, node_charges) ambient"
        )
    return exchange


def class_S_A1_bps_ambient(n_punctures: int,
                           marks_list: Sequence[int]) -> dict:
    """Ambient (unimodular) lift of :func:`class_S_A1_bps` .

    Returns the same theory in a unimodular symplectic ambient
    Z^{2 N_eff}  with pairing  block-diag([[0,1],[-1,0]])  instead of
    the FST-derived non-unimodular sublattice with standard-basis
    nodes.  Output dict keys:

        pairing        ambient lattice pairing  Lambda  (rank x rank)
        node_charges   each FST internal edge as a charge in the
                       ambient lattice
        spec           the same  S  factors, expressed in the ambient
        exchange       <node_i, node_j>  computed from  Lambda  --
                       reproduces  class_S_A1_bps(...)["exchange"]
        n_nodes        len(node_charges)
        start, cuts    same as  class_S_A1_bps

    PROTOTYPE -- only the pure-SU(2) endpoint  (Nf=0/Nf=0, N=1, no
    cuts)  is supported so far.  Other configurations raise
    ``NotImplementedError``.
    """
    from legacy.class_s_a1_bps import class_S_A1_bps as _class_S_A1_bps_legacy
    flat = _class_S_A1_bps_legacy(n_punctures, marks_list)
    N, left_nf, right_nf = flat["start"]
    cuts = flat["cuts"]

    if (N, left_nf, right_nf, len(cuts)) == (1, 0, 0, 0):
        # Pure SU(2): Gamma = Z^2, standard symplectic, nodes are the
        # two BPS charges of pure SU(2) in the SW chamber.  See
        pairing = ((0, 1), (-1, 0))
        node_charges = ((1, 0), (-1, 2))
        spec = ((1, 0), (-1, 2))
    elif (left_nf == 2 and right_nf == 2 and len(cuts) == 0):
        # n-punctured sphere (Nf=2/Nf=2 caps, no boundaries).
        #
        # Strategy: build the SW chamber natively (Kronecker-2 gauge
        # pairs, matter / bifund as composites; gauge / flavour split
        # is explicit in the slot layout) with the diamond flavour
        # convention  (1,1) / (1,-1)  for each Nf=2 cap, then
        # transport across the matter-block mutation per side to
        # land in the FST chamber.
        #
        # Slot layout for  N = n - 3  gauge factors:
        #
        #   factor 1:  gauge slots (start_1, start_1+1)  +  Z^2 left
        #              flavour slots (start_1+2, start_1+3)
        #   factor i  (1 < i < N):   gauge slots only (2 slots)
        #   factor N:  gauge slots  +  Z^2 right flavour slots
        #   then  N - 1  bifund flavour slots in ker B at the tail.
        #
        # Total rank = 2N + 4 + (N - 1) = 3N + 3 = 3n - 6  as required.
        #
        # Each SU(2) gauge factor is the SW Kronecker-2 chamber:
        #   gamma_i = e_{start_i},   beta_i = -e_{start_i} + 2 e_{start_i+1}
        #   <gamma_i, beta_i> = 2 .
        # Lowest weight of the fundamental of SU(2) is  (0, -1)  in
        # the gauge block (per SUN_Nf's convention).  Each Nf=2 cap
        # then has matter nodes at
        #   m_+  =  lowest_weight + (1, 1)_{cap flavour}
        #   m_-  =  lowest_weight + (1, -1)_{cap flavour}
        # with both matter charges sharing the cap's Z^2 flavour
        # sublattice (slots in ker B), forming a diamond pattern at
        # +/- 45 degrees from the flavour-slot axes.
        sw = _build_sw_chamber_n_sphere(N)
        Q_amb = BPSQuiver.from_pairing(sw["nodes"], sw["pairing"])
        spec_amb: tuple[tuple[int, ...], ...] = tuple(sw["spec"])
        # Sanity: SW chamber spec is a valid negating sequence on
        # itself before we mutate.
        ok, _ = Q_amb.verify_spectrum_generator(list(spec_amb))
        if not ok:
            raise RuntimeError(
                f"native SW-chamber construction (N={N}) does not "
                f"verify as a negating sequence; ambient lift is "
                f"inconsistent."
            )

        # Transport across one matter-block mutation per Nf=2 cap.
        # Block size  = 2 * N_factor - 1 = 3  for SU(2) Nf=2.
        B_target = [list(r) for r in flat["exchange"]]
        n_target = len(B_target)
        candidate_block_sizes = (3, 1, 5)

        def _dfs(Q, sp, mutations_left):
            if mutations_left == 0:
                sigma = _find_node_permutation(B_target, Q.exchange)
                if sigma is None:
                    return None
                return Q, sp, sigma
            for k in range(Q.n_nodes):
                charge_k = tuple(Q.charges[k])
                if not any(tuple(g) == charge_k for g in sp):
                    continue
                for bs in candidate_block_sizes:
                    res = _mutate_matter_block_ambient(Q, sp, k, bs)
                    if res is None:
                        continue
                    Q1, sp1 = res
                    rec = _dfs(Q1, sp1, mutations_left - 1)
                    if rec is not None:
                        return rec
            return None

        result = _dfs(Q_amb, spec_amb, 2)
        if result is None:
            raise RuntimeError(
                f"SW->FST mutation transport failed for "
                f"N={N} (n_punctures={n_punctures}); the diamond-"
                f"flavour SW chamber did not reach an FST-aligned "
                f"chamber within 2 matter-block mutations."
            )
        Q_after, spec_after, sigma = result

        permuted_charges = tuple(
            tuple(Q_after.charges[sigma[j]]) for j in range(n_target)
        )
        pairing = tuple(tuple(int(x) for x in row) for row in sw["pairing"])
        node_charges = permuted_charges
        spec = tuple(spec_after)
    else:
        raise NotImplementedError(
            f"ambient lift not yet implemented for "
            f"(N={N}, left_nf={left_nf}, right_nf={right_nf}, "
            f"|cuts|={len(cuts)}); prototype currently covers only "
            f"the pure-SU(2) endpoint (1, 0, 0, no cuts) and the "
            f"n-punctured sphere (Nf=2/Nf=2, no cuts)."
        )

    exchange = _verify_ambient(pairing, node_charges, spec)
    if exchange != flat["exchange"]:
        raise RuntimeError(
            f"ambient-lift exchange {exchange} does not match flat "
            f"FST exchange {flat['exchange']}"
        )

    # Per-puncture flavour Cartan basis (only for the n-puncture
    # sphere; pure SU(2) has no flavour).
    cartans: list[tuple[str, tuple[int, ...]]] | None
    if (left_nf == 2 and right_nf == 2 and len(cuts) == 0):
        cartans = n_sphere_puncture_cartans(N)
    else:
        cartans = None

    return {
        "pairing": pairing,
        "node_charges": node_charges,
        "spec": spec,
        "exchange": exchange,
        "n_nodes": len(node_charges),
        "start": flat["start"],
        "cuts": cuts,
        "puncture_cartans": cartans,
    }
