"""Simple multicurves on S^2_{0,n} in normal position, and the
combinatorial diagram calculus needed for the intrinsic skein product.

Mathematical setting
--------------------
Fix an ideal triangulation `Delta` of the closed n-punctured sphere
(`triangulation.Triangulation`, Triangle-based constructor: every edge
joins two distinct punctures, every triangle has three distinct
vertices).  A *simple multicurve* is an isotopy class of embedded,
unoriented, closed 1-manifolds in the punctured sphere with no
contractible component.  Classical fact (normal-curve theory for ideal
triangulations): isotopy classes of simple multicurves correspond
bijectively to **admissible coordinate vectors**

    a in Z_{>=0}^{E(Delta)},   a_e = geometric intersection number with e,

where admissibility means, per triangle with edge intersections
(x, y, z): parity `x + y + z` even and the triangle inequalities
`x <= y + z` (cyclically).  Inside each triangle the multicurve is a
union of disjoint *corner arcs*; the corner counts are
`m_k = (x_{k+1} + x_{k+2} - x_k)/2`, and arcs of the same corner are
parallel (nested around the corner vertex).

Peripheral components (small loops around a puncture `p`) are normal,
with coordinates `f_p = Triangulation.puncture_flavour_charge(p)`.

This module provides:

* admissibility / corner counts (`is_admissible`, `corner_counts`);
* point-indexed normal arcs (`normal_arcs`) -- each intersection point
  with an edge gets an index along the edge, and each arc records its
  two (edge, point) endpoints;
* `MatchingDiagram` -- a crossingless diagram that need NOT be normal
  (arcs may "return" to the edge they started from).  These arise as
  Kauffman smoothings of stacked multicurves;
* `normalize` -- tighten a `MatchingDiagram` to normal position by
  iterated returning-arc (spike) removal, counting the contractible
  loops that die in the process;
* component extraction (`components_of_coords`) and the peripheral /
  core split of a normal multicurve.

Self-contained: depends only on `triangulation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from triangulation import Triangulation


Coords = tuple[int, ...]


# ---------------------------------------------------------------------
# Admissibility and corner counts
# ---------------------------------------------------------------------


def is_admissible(t: Triangulation, coords: Sequence[int]) -> bool:
    """Whether `coords` are the normal coordinates of a simple multicurve."""
    if len(coords) != t.n_edges:
        return False
    if any((not isinstance(x, int)) or x < 0 for x in coords):
        return False
    for te in t.triangle_edges:
        x = [coords[e] for e in te]
        if (x[0] + x[1] + x[2]) % 2 != 0:
            return False
        for k in range(3):
            if x[k] > x[(k + 1) % 3] + x[(k + 2) % 3]:
                return False
    return True


def corner_counts(t: Triangulation, coords: Sequence[int]) -> list[tuple[int, int, int]]:
    """Per triangle, the corner-arc counts `(m_0, m_1, m_2)`.

    `m_k` is the number of arcs joining the edges at ccw positions
    `k+1` and `k+2`, i.e. the arcs cutting off the corner at the vertex
    shared by those two edges (= `verts[(k+2) % 3]` for Triangle-based
    triangulations).
    """
    out = []
    for te in t.triangle_edges:
        x = [coords[e] for e in te]
        m = []
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            v = x[i] + x[j] - x[k]
            if v % 2 != 0 or v < 0:
                raise ValueError(
                    f"coords {tuple(coords)} not admissible in triangle {te}"
                )
            m.append(v // 2)
        out.append(tuple(m))
    return out


# ---------------------------------------------------------------------
# Point-indexed arcs of a normal multicurve
# ---------------------------------------------------------------------
#
# Conventions.
#
# * Points on edge `e` are indexed `0 .. a_e - 1` along the edge's
#   canonical orientation `e.u -> e.v` (u <= v as stored).
#
# * In triangle `ti` with ccw vertex tuple (w_0, w_1, w_2) and ccw edges
#   (E_0, E_1, E_2) (E_i joins w_i to w_{i+1}), the corner at vertex
#   w_{k+2} carries the `m_k` arcs joining E_{k+1} and E_{k+2}.
#   Their nesting depth is `d = 0 .. m_k - 1`, with `d = 0` the arc
#   closest to the vertex.  On each of the two edges, the corner's arcs
#   occupy the `m_k` points nearest the corner vertex, depth d at the
#   d-th point from that vertex's end of the edge.


@dataclass(frozen=True)
class PointedArc:
    """One normal arc inside a triangle, with point-indexed endpoints.

    `corner_pos` is `k` (the arc joins ccw edge positions k+1, k+2);
    `depth` is the nesting depth (0 = innermost, nearest the vertex).
    `end_a = (edge_id, point)` lies on the ccw-earlier edge (position
    k+1) and `end_b` on the ccw-later edge (position k+2).
    """
    triangle: int
    corner_pos: int
    depth: int
    end_a: tuple[int, int]
    end_b: tuple[int, int]


def _point_from_vertex_end(
    t: Triangulation, ti: int, pos: int, vertex_is_start: bool,
    depth: int, coords: Sequence[int],
) -> int:
    """Global point index of the `depth`-th point from one end of the
    edge at ccw position `pos` of triangle `ti`.

    `vertex_is_start=True` means: count from the end of the edge
    segment at the triangle-ccw START of the slot's traversal;
    otherwise from the traversal END.  Points are indexed along the
    edge's canonical direction, so the answer only needs the slot's
    orientation bit (`t.slot_forward[ti][pos]`) — no vertex labels.
    This is what makes self-loops and multi-edges representable: the
    two ends of an edge SEGMENT are always distinct, even when both
    lie at the same puncture (id-based edge refactor, 2026-07-07).
    """
    e_id = t.triangle_edges[ti][pos]
    n_pts = coords[e_id]
    at_canonical_start = (t.slot_forward[ti][pos] == vertex_is_start)
    return depth if at_canonical_start else n_pts - 1 - depth


def normal_arcs(t: Triangulation, coords: Sequence[int]) -> list[list[PointedArc]]:
    """The point-indexed normal arcs, one list per triangle.

    Works on both construction paths (strict Triangle-based and
    `from_edge_data` with self-loops / multi-edges) — the point
    indexing consumes only `slot_forward` orientation bits.
    """
    coords = tuple(coords)
    if not is_admissible(t, coords):
        raise ValueError(f"coords {coords} are not admissible")
    corners = corner_counts(t, coords)
    out: list[list[PointedArc]] = []
    for ti in range(t.n_triangles):
        arcs: list[PointedArc] = []
        m = corners[ti]
        for k in range(3):
            pos_a = (k + 1) % 3   # ccw-earlier edge of the corner
            pos_b = (k + 2) % 3   # ccw-later edge of the corner
            # The corner vertex is w_{k+2}: the END vertex of edge at
            # pos_a, and the START vertex of edge at pos_b.
            for d in range(m[k]):
                pa = _point_from_vertex_end(t, ti, pos_a, False, d, coords)
                pb = _point_from_vertex_end(t, ti, pos_b, True, d, coords)
                arcs.append(PointedArc(
                    triangle=ti, corner_pos=k, depth=d,
                    end_a=(t.triangle_edges[ti][pos_a], pa),
                    end_b=(t.triangle_edges[ti][pos_b], pb),
                ))
        out.append(arcs)
    return out


# ---------------------------------------------------------------------
# Crossingless diagrams (not necessarily normal) and normalization
# ---------------------------------------------------------------------


class MatchingDiagram:
    """A crossingless closed-curve diagram on the triangulated sphere.

    Data: for each triangle, a perfect matching on its boundary points;
    boundary points of triangle `ti` are pairs `(edge_id, point_index)`
    where `edge_id` ranges over the triangle's three edges and
    `point_index` over `0 .. n_points[edge_id] - 1`.  Each edge carries
    the same point set in both adjacent triangles, so the matchings
    glue into closed curves.

    Unlike a normal multicurve, a matching arc may have both endpoints
    on the same edge (a *returning arc*).  `normalize()` removes those.

    The diagram is assumed embedded (matchings non-crossing inside each
    triangle); this is guaranteed by construction for Kauffman
    smoothings of stacked embedded multicurves and is not re-checked.
    """

    def __init__(
        self,
        t: Triangulation,
        n_points: Sequence[int],
        arcs_per_triangle: list[list[tuple[tuple[int, int], tuple[int, int]]]],
    ):
        self.t = t
        self.n_points = list(n_points)
        # normalize arc storage: frozenset endpoints per arc
        self.arcs: list[list[tuple[tuple[int, int], tuple[int, int]]]] = [
            [tuple(sorted(a)) for a in tri_arcs] for tri_arcs in arcs_per_triangle
        ]

    @classmethod
    def from_normal(cls, t: Triangulation, coords: Sequence[int]) -> "MatchingDiagram":
        arcs = normal_arcs(t, coords)
        return cls(
            t,
            list(coords),
            [[(a.end_a, a.end_b) for a in tri] for tri in arcs],
        )

    # -- normalization ------------------------------------------------

    def normalize(self) -> tuple[Coords, int]:
        """Tighten to normal position.

        Returns `(coords, n_trivial)` where `coords` are the normal
        coordinates of the tightened multicurve and `n_trivial` is the
        number of contractible components that died during tightening
        (each contributes one factor of the trivial-loop value to a
        skein computation).

        Algorithm: repeatedly find a *returning arc* -- an arc of some
        triangle `T` with both endpoints on the same edge `e`, with no
        other returning arc nested inside it on `e` -- and isotope it
        across `e` into the neighbouring triangle `T'`, deleting its two
        points from `e` and merging the two `T'`-arcs at those points
        into one.  If those two `T'`-arcs coincide (the returning arc is
        capped by a single arc), the component is a closed loop inside
        the disk `T cup_e T'`: it is contractible and is removed,
        incrementing `n_trivial`.  Each step lowers `sum(n_points)` by
        2, so the loop terminates; a diagram with no returning arcs is
        normal, and its coordinates are `n_points`.
        """
        t = self.t
        n_points = list(self.n_points)
        arcs = [list(tri) for tri in self.arcs]
        n_trivial = 0

        def find_returning() -> tuple[int, int] | None:
            """(triangle, arc index) of an innermost returning arc."""
            best = None
            for ti in range(t.n_triangles):
                for ai, (p, q) in enumerate(arcs[ti]):
                    if p[0] != q[0]:
                        continue
                    # innermost on its edge: no other returning arc of
                    # the same triangle/edge strictly inside [p, q]
                    lo, hi = min(p[1], q[1]), max(p[1], q[1])
                    inner = False
                    for bi, (r, s) in enumerate(arcs[ti]):
                        if bi == ai or r[0] != p[0] or s[0] != p[0]:
                            continue
                        rlo, rhi = min(r[1], s[1]), max(r[1], s[1])
                        if lo < rlo and rhi < hi:
                            inner = True
                            break
                    if not inner:
                        # embeddedness => points strictly inside [lo,hi]
                        # on edge e all belong to arcs nested inside,
                        # which are returning arcs of the SAME triangle
                        # (any arc with exactly one endpoint in (lo,hi)
                        # would cross).  With none returning, [lo,hi]
                        # is an innermost cap: hi == lo + 1.
                        best = (ti, ai)
                        return best
            return best

        while True:
            found = find_returning()
            if found is None:
                break
            ti, ai = found
            (e, p1), (_, p2) = arcs[ti][ai]
            lo, hi = min(p1, p2), max(p1, p2)
            if hi != lo + 1:
                raise AssertionError(
                    "innermost returning arc endpoints not adjacent; "
                    "diagram was not embedded"
                )
            # neighbour triangle across e
            inc = t.edge_incidences[e]
            others = [pair[0] for pair in inc if pair[0] != ti]
            if len(inc) != 2:
                raise AssertionError("edge not shared by two triangles")
            tj = inc[0][0] if inc[0][0] != ti else inc[1][0]
            if tj == ti:
                raise AssertionError("self-glued edge unsupported")
            # the two T'-arcs at (e, lo), (e, hi)
            def arc_at(tk: int, end: tuple[int, int]) -> int:
                hits = [bi for bi, (r, s) in enumerate(arcs[tk]) if r == end or s == end]
                if len(hits) != 1:
                    raise AssertionError(f"point {end} not matched exactly once in triangle {tk}")
                return hits[0]

            bj1 = arc_at(tj, (e, lo))
            bj2 = arc_at(tj, (e, hi))
            if bj1 == bj2:
                # capped: closed contractible loop dies
                n_trivial += 1
                del arcs[ti][ai]
                del arcs[tj][bj1]
            else:
                r1, s1 = arcs[tj][bj1]
                other1 = s1 if r1 == (e, lo) else r1
                r2, s2 = arcs[tj][bj2]
                other2 = s2 if r2 == (e, hi) else r2
                # remove the three old arcs, add the merged T'-arc
                for bi in sorted([bj1, bj2], reverse=True):
                    del arcs[tj][bi]
                del arcs[ti][ai]
                arcs[tj].append(tuple(sorted((other1, other2))))
            # delete points lo, hi from edge e; shift higher indices by 2
            n_points[e] -= 2

            def shift(end: tuple[int, int]) -> tuple[int, int]:
                ee, pp = end
                if ee != e:
                    return end
                if pp > hi:
                    return (ee, pp - 2)
                if pp < lo:
                    return end
                raise AssertionError("deleted point still referenced")

            for tk in t.triangles_at_edge(e):
                arcs[tk] = [tuple(sorted((shift(r), shift(s)))) for (r, s) in arcs[tk]]

        coords = tuple(n_points)
        if not is_admissible(t, coords):
            raise AssertionError(
                f"normalized diagram has inadmissible coords {coords}"
            )
        # Re-derive as a sanity check: a returning-arc-free matching on
        # admissible coords must BE the normal matching (corner counts
        # are forced), so we can simply return the coords.
        self_check = corner_counts(t, coords)
        for ti in range(t.n_triangles):
            if len(arcs[ti]) != sum(self_check[ti]):
                raise AssertionError("arc count mismatch after normalization")
        return coords, n_trivial

    # -- component walking --------------------------------------------

    def components(self) -> list[list[tuple[int, int]]]:
        """Connected components as cyclic lists of (edge, point) crossings.

        Each (edge, point) appears in exactly two triangle matchings
        (the two sides); walking alternates sides.
        """
        t = self.t
        # map (triangle, end) -> partner end within that triangle
        partner: dict[tuple[int, tuple[int, int]], tuple[int, int]] = {}
        for ti in range(t.n_triangles):
            for (r, s) in self.arcs[ti]:
                partner[(ti, r)] = s
                partner[(ti, s)] = r
        # map (edge, point) -> the two triangles
        sides: dict[tuple[int, int], list[int]] = {}
        for (ti, end), _ in partner.items():
            sides.setdefault(end, []).append(ti)
        for end, ts in sides.items():
            if len(ts) != 2:
                raise AssertionError(f"point {end} has {len(ts)} sides (expected 2)")
        comps: list[list[tuple[int, int]]] = []
        visited: set[tuple[int, int]] = set()
        for start in sorted(sides):
            if start in visited:
                continue
            comp = []
            end = start
            ti = sides[start][0]
            while True:
                comp.append(end)
                visited.add(end)
                nxt = partner[(ti, end)]
                # cross to the other side of nxt's edge
                ta, tb = sides[nxt]
                ti = tb if ta == ti else ta
                end = nxt
                if end == start:
                    break
            comps.append(comp)
        return comps


# ---------------------------------------------------------------------
# Components of a normal multicurve; peripheral / core split
# ---------------------------------------------------------------------


def components_of_coords(
    t: Triangulation, coords: Sequence[int],
) -> list[Coords]:
    """Normal coordinates of each connected component of the multicurve.

    Components of a normal multicurve are themselves normal, and the
    coordinates add: `coords == sum(components)`.
    """
    diagram = MatchingDiagram.from_normal(t, coords)
    comps = diagram.components()
    out: list[Coords] = []
    for comp in comps:
        v = [0] * t.n_edges
        for (e, _p) in comp:
            v[e] += 1
        out.append(tuple(v))
    return out


def peripheral_split(
    t: Triangulation, coords: Sequence[int],
) -> tuple[Coords, tuple[int, ...]]:
    """Split a normal multicurve into (core coords, peripheral counts).

    Returns `(core, k)` where `k[p]` counts the peripheral components
    around puncture `p` and `core` is the coordinate sum of the
    non-peripheral components.
    """
    fps = [t.puncture_flavour_charge(p) for p in range(t.n_punctures)]
    k = [0] * t.n_punctures
    core = [0] * t.n_edges
    for comp in components_of_coords(t, coords):
        matched = None
        for p, fp in enumerate(fps):
            if comp == fp:
                matched = p
                break
        if matched is None:
            for i, x in enumerate(comp):
                core[i] += x
        else:
            k[matched] += 1
    return tuple(core), tuple(k)
