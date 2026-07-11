"""Curve realization from a tropical charge.

Input:
    triangulation : an ideal triangulation of S^2_{0,n}.
    a             : tropical charge, a list/tuple of integers indexed by
                    triangulation.edges (canonical edge-index order).
                    |a_e| is the geometric intersection count of the
                    realized curve with edge e; the sign of a_e encodes
                    the twist direction across e (used in elevation
                    ordering when an edge has multiple intersections).

Output:
    For each triangle T of the triangulation, the list of arcs of
    gamma_a within T.  Each arc is a `TriangleArc` carrying:
       * its type (alpha / beta / gamma) -- determined by which two
         edges of T it connects, in T's ccw edge order;
       * the global edge id at the ccw-later endpoint (used to read
         the `eps` state in the phi-map);
       * the global edge id at the ccw-earlier endpoint (`eps'`).

Scope (asserted):
    * Each triangle has at most one arc (m_ij^T in {0, 1} for all
      pairs).  Multi-arc triangles are flagged for future work.
    * The triangle inequality and parity checks must hold.
    * No "all-zero" / peripheral curves are returned (a zero charge
      yields no arcs in any triangle and an empty curve, which is the
      empty tangle / vacuum element of the skein algebra).

Self-contained: depends only on triangulation and phi_map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from phi_map import ALPHA, BETA, GAMMA
from triangulation import Triangulation


@dataclass(frozen=True)
class TriangleArc:
    """A single arc of the realized curve within one triangle.

    `arc_type` is `alpha`, `beta`, or `gamma` (cf. phi_map).  In the
    ccw edge cycle of the triangle (a, b, c) at positions (0, 1, 2):
        alpha connects positions 1 and 2 (b and c)
        beta  connects positions 2 and 0 (c and a)
        gamma connects positions 0 and 1 (a and b)

    State convention: given a state
    eps at the arc's ccw-LATER endpoint and eps' at the ccw-EARLIER
    endpoint, phi(arc_type, eps, eps') gives the un-Weyl-normalized
    contribution.  The two `edge_at_*` fields below record which
    triangulation edge (global id) carries each of those states.

    For each arc type the assignment is fixed:
        alpha : ccw-later = c (pos 2), ccw-earlier = b (pos 1)
        beta  : ccw-later = a (pos 0), ccw-earlier = c (pos 2)
        gamma : ccw-later = b (pos 1), ccw-earlier = a (pos 0)
    """
    arc_type: str
    edge_at_eps: int        # global edge id at the ccw-later endpoint
    edge_at_eps_prime: int  # global edge id at the ccw-earlier endpoint


# Map from {ccw position pair} to arc type.  Position pairs are
# unordered ({i, j}), so we key by a frozenset.
_ARC_TYPE_BY_POSITIONS: dict[frozenset[int], str] = {
    frozenset({1, 2}): ALPHA,   # b, c -> alpha
    frozenset({2, 0}): BETA,    # c, a -> beta
    frozenset({0, 1}): GAMMA,   # a, b -> gamma
}

# For each arc type, the (ccw-later position, ccw-earlier position).
_CCW_LATER_FIRST: dict[str, tuple[int, int]] = {
    ALPHA: (2, 1),  # c later, b earlier
    BETA:  (0, 2),  # a later, c earlier
    GAMMA: (1, 0),  # b later, a earlier
}


class RealizationError(ValueError):
    """Raised when a tropical charge does not realize a single simple
    closed curve under our current scope (single-arc-per-triangle).
    """


def realize_curve(
    triangulation: Triangulation,
    charge: Sequence[int],
) -> list[list[TriangleArc]]:
    """Build the per-triangle arc decomposition of gamma_a.

    Returns a list of length `n_triangles`, with the i-th entry being
    the list of arcs of gamma_a inside triangle i (in the
    triangulation's triangle order).

    Validates parity, triangle inequalities, and the single-arc-per-
    triangle scope.  Raises RealizationError on violation.
    """
    n_edges = triangulation.n_edges
    if len(charge) != n_edges:
        raise RealizationError(
            f"charge has length {len(charge)}, expected {n_edges}"
        )

    arcs_per_triangle: list[list[TriangleArc]] = []
    for ti, edges_ccw in enumerate(triangulation.triangle_edges):
        a = [abs(charge[e]) for e in edges_ccw]
        # parity
        if (a[0] + a[1] + a[2]) % 2 != 0:
            raise RealizationError(
                f"triangle {ti}: |a_0|+|a_1|+|a_2| = {a[0]+a[1]+a[2]} is odd; "
                "parity check fails (Section 9.1)"
            )
        # triangle inequalities
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            if a[k] > a[i] + a[j]:
                raise RealizationError(
                    f"triangle {ti}: |a_{k}|={a[k]} > |a_{i}|+|a_{j}|={a[i]+a[j]}; "
                    "triangle inequality fails (Section 9.1)"
                )

        # m_ij^T = (|a_i| + |a_j| - |a_k|) / 2, for {i,j,k} = {0,1,2}
        m = [0, 0, 0]
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            m[k] = (a[i] + a[j] - a[k]) // 2
        # m[k] is the number of arcs OPPOSITE position k, i.e. connecting
        # positions (k+1, k+2).  In ccw notation:
        #     m[0] = arcs opposite a -> alpha-type arcs (b<->c)
        #     m[1] = arcs opposite b -> beta-type  arcs (c<->a)
        #     m[2] = arcs opposite c -> gamma-type arcs (a<->b)
        total = m[0] + m[1] + m[2]
        if total > 1:
            raise RealizationError(
                f"triangle {ti}: total arc count m={total} > 1; "
                "multi-arc-per-triangle is out of scope (Section 9.4 / "
                "example doc Section 8 caveat)"
            )

        triangle_arcs: list[TriangleArc] = []
        for k in range(3):
            if m[k] == 0:
                continue
            # arc opposite position k connects positions (k+1) and (k+2)
            i, j = (k + 1) % 3, (k + 2) % 3
            arc_type = _ARC_TYPE_BY_POSITIONS[frozenset({i, j})]
            later_pos, earlier_pos = _CCW_LATER_FIRST[arc_type]
            triangle_arcs.append(
                TriangleArc(
                    arc_type=arc_type,
                    edge_at_eps=edges_ccw[later_pos],
                    edge_at_eps_prime=edges_ccw[earlier_pos],
                )
            )
        arcs_per_triangle.append(triangle_arcs)

    # Global consistency: total number of arcs equals N = sum |a_e|.
    # (Per triangle, sum of m_ij = (|a_T0|+|a_T1|+|a_T2|)/2.  Summing
    # across triangles, each |a_e| is counted twice (each edge in 2
    # triangles), giving total = sum_e |a_e| = N.)
    total_arcs = sum(len(tlist) for tlist in arcs_per_triangle)
    n_intersections = sum(abs(charge[e]) for e in range(n_edges))
    if total_arcs != n_intersections:
        raise RealizationError(
            f"global consistency check failed: total #arcs = {total_arcs} "
            f"but sum |a_e| = {n_intersections}; "
            "the realized curve does not close up "
            "(usually a triangulation/charge mismatch)"
        )

    return arcs_per_triangle


def crossed_edges(charge: Sequence[int]) -> list[int]:
    """Indices of edges crossed by gamma_a, i.e. those with |a_e| > 0."""
    return [i for i, v in enumerate(charge) if v != 0]
