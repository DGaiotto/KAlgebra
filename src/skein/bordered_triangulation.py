"""Bordered triangulations: ideal triangles glued along SOME edge
pairs; unglued sides are boundary edges (single incidence).

This is the combinatorial substrate of the stated-skein polygon engine
(`stated_polygon`) and of the cutting /
gluing RG story: a `BorderedTriangulation` knows its triangles,
its internal (glued) edges, its boundary sides in ccw order, and the
dual graph; for triangulated polygons the dual graph is a tree, so
embedded arcs between boundary sides route uniquely.

Side roles within each triangle follow the `stated_triangle`
convention: roles 0, 1, 2 = edges a, b, c in ccw order (vertices
v0, v1, v2 ccw; a = v0->v1, b = v1->v2, c = v2->v0).

The fan polygon: vertices 0..n-1 ccw; triangles T_i = (0, i, i+1) for
i = 1..n-2 (stored 0-indexed as triangle i-1); internal edges (0, i)
glue T_{i-1}.c to T_i.a (i = 2..n-2); polygon side j = (j, j+1 mod n):
side 0 = T_1.a, side i = T_i.b (1 <= i <= n-2), side n-1 = T_{n-2}.c.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Side roles
ROLE_A, ROLE_B, ROLE_C = 0, 1, 2


@dataclass(frozen=True)
class BorderedTriangulation:
    """Triangles + gluing.

    triangle_verts   tuple of (v0, v1, v2) ccw vertex triples
    gluings          tuple of ((tri, role), (tri, role)) pairs, each
                     slot appearing at most once
    boundary_sides   tuple of (tri, role) slots, ccw around the
                     boundary (supplied by the constructor; for the
                     fan polygon, polygon-side order)
    """

    triangle_verts: tuple
    gluings: tuple
    boundary_sides: tuple

    def __post_init__(self):
        seen = set()
        for pair in self.gluings:
            (t1, _r1), (t2, _r2) = pair
            if t1 == t2:
                # Gluing two sides of ONE triangle is legitimate in Le's
                # theory (it produces self-loop/self-folded
                # configurations), but this engine's routing and
                # stated-polygon machinery assume it never happens —
                # honest-fail rather than silently corrupt.
                raise NotImplementedError(
                    f"gluing two sides of the same triangle {t1} "
                    "(self-gluing) is not supported by this engine"
                )
            for slot in pair:
                if slot in seen:
                    raise ValueError(f"slot {slot} glued twice")
                seen.add(slot)
        for slot in self.boundary_sides:
            if slot in seen:
                raise ValueError(f"boundary slot {slot} is glued")
            seen.add(slot)
        n_slots = 3 * len(self.triangle_verts)
        if len(seen) != n_slots:
            raise ValueError(
                f"{n_slots - len(seen)} sides neither glued nor boundary"
            )

    @property
    def n_triangles(self) -> int:
        return len(self.triangle_verts)

    @property
    def n_boundary(self) -> int:
        return len(self.boundary_sides)

    # -- edge numbering + FST exchange matrix ---------------------------
    #
    # A bordered triangulation has TWO kinds of edges: interior (glued)
    # edges, shared by their two slots, and boundary edges (the unglued
    # marked intervals of an irregular puncture), one slot each.  The FST
    # exchange matrix sigma is computed over ALL of them by the same
    # ccw-triangle rule as `Triangulation.sigma()`; the MUTABLE block
    # (interior x interior) is the BPS quiver, the boundary edges are
    # frozen.  This is the substrate of the "not fully gluing all edges"
    # construction (irregular-puncture extension).

    def edge_numbering(self):
        """Assign a global edge id to every edge: interior (glued) edges
        first (in `gluings` order), then boundary edges (in
        `boundary_sides` order).  Returns
        ``(slot_to_edge, internal_ids, boundary_ids)`` where
        ``slot_to_edge`` maps every ``(tri, role)`` slot to its edge id.
        """
        slot_to_edge: dict[tuple[int, int], int] = {}
        internal_ids: list[int] = []
        boundary_ids: list[int] = []
        nid = 0
        for a, b in self.gluings:
            slot_to_edge[a] = nid
            slot_to_edge[b] = nid
            internal_ids.append(nid)
            nid += 1
        for slot in self.boundary_sides:
            slot_to_edge[slot] = nid
            boundary_ids.append(nid)
            nid += 1
        return slot_to_edge, internal_ids, boundary_ids

    @property
    def n_edges(self) -> int:
        return len(self.gluings) + len(self.boundary_sides)

    @property
    def internal_edge_ids(self) -> list[int]:
        return self.edge_numbering()[1]

    @property
    def boundary_edge_ids(self) -> list[int]:
        return self.edge_numbering()[2]

    def sigma(self) -> list[list[int]]:
        """The FST exchange matrix sigma over ALL edges (interior +
        boundary), an ``n_edges x n_edges`` Z-matrix, by the same ccw rule
        as `Triangulation.sigma()`:

            sigma(e, f) = (# triangles in which f follows e ccw)
                        - (# triangles in which e follows f ccw).

        Edge ids follow `edge_numbering()` (interior first, boundary
        last), so the top-left ``len(gluings) x len(gluings)`` block is the
        MUTABLE quiver and the rest are the frozen boundary directions.
        """
        slot_to_edge, _, _ = self.edge_numbering()
        n = self.n_edges
        s = [[0] * n for _ in range(n)]
        for tri in range(self.n_triangles):
            for pos in range(3):
                e_i = slot_to_edge[(tri, pos)]
                e_j = slot_to_edge[(tri, (pos + 1) % 3)]
                s[e_i][e_j] += 1
                s[e_j][e_i] -= 1
        return s

    def mutable_block(self) -> list[list[int]]:
        """The interior x interior principal block of `sigma()` -- the BPS
        Dirac pairing of the canonical algebra (boundary edges frozen)."""
        s = self.sigma()
        ids = self.internal_edge_ids
        return [[s[i][j] for j in ids] for i in ids]

    def glued_partner(self, tri: int, role: int):
        """The (tri, role) glued to this slot, or None if boundary."""
        for x, y in self.gluings:
            if x == (tri, role):
                return y
            if y == (tri, role):
                return x
        return None

    def dual_neighbours(self, tri: int):
        """(role, other_tri, other_role) for each glued side of tri."""
        out = []
        for role in (ROLE_A, ROLE_B, ROLE_C):
            p = self.glued_partner(tri, role)
            if p is not None:
                out.append((role, p[0], p[1]))
        return out

    def dual_path(self, tri_from: int, tri_to: int):
        """BFS path in the dual graph: list of (tri, in_role, out_role)
        with in_role None at the start and out_role None at the end.
        For trees (polygons) the path is unique."""
        if tri_from == tri_to:
            return [(tri_from, None, None)]
        prev = {tri_from: None}
        queue = [tri_from]
        while queue:
            t = queue.pop(0)
            if t == tri_to:
                break
            for role, t2, role2 in self.dual_neighbours(t):
                if t2 not in prev:
                    prev[t2] = (t, role, role2)
                    queue.append(t2)
        if tri_to not in prev:
            raise ValueError(f"triangles {tri_from}, {tri_to} not connected")
        # reconstruct
        chain = []
        t = tri_to
        in_role = None
        while True:
            p = prev[t]
            if p is None:
                chain.append((t, None, in_role))
                break
            t_prev, role_prev, role_here = p
            chain.append((t, role_here, in_role))
            in_role = role_prev
            t = t_prev
        chain.reverse()
        # chain entries are (tri, in_role, out_role) with the first
        # in_role None and the last out_role None
        return chain

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_closed(cls, tri, cut_edge_ids=()) -> "BorderedTriangulation":
        """Bordered triangulation from a CLOSED Triangle-based
        ``Triangulation``, optionally **cutting** (un-gluing) the FST edges
        with the given ids (in ``tri.edges`` order): each cut edge's two
        slots become boundary edges.

        This is the "not fully gluing all edges" operation -- cutting an
        FST edge turns a regular gluing into a boundary marked interval,
        i.e. creates (part of) an irregular puncture.  Structurally it
        deletes that edge from the mutable quiver (`mutable_block()` is the
        closed `tri.sigma()` with the cut rows/cols removed), so the
        canonical algebra is the node-drop RG IR theory.

        Requires the Triangle-based representation (``tri.triangles``
        non-empty); the ``from_edge_data`` path (multi-edges / self-loops)
        is not handled here.
        """
        if not getattr(tri, "triangles", None):
            raise NotImplementedError(
                "from_closed needs a Triangle-based Triangulation "
                "(multi-edge / self-loop from_edge_data charts unsupported)."
            )
        triangle_verts = tuple(t.verts for t in tri.triangles)
        cut = set(cut_edge_ids)
        gluings, boundary = [], []
        for eid, inc in enumerate(tri.edge_incidences):
            (t1, p1), (t2, p2) = inc[0], inc[1]
            if eid in cut:
                boundary.extend([(t1, p1), (t2, p2)])
            else:
                gluings.append(((t1, p1), (t2, p2)))
        return cls(triangle_verts, tuple(gluings), tuple(boundary))

    @classmethod
    def fan_polygon(cls, n: int) -> "BorderedTriangulation":
        """The triangulated n-gon (n >= 3), fan at vertex 0."""
        if n < 3:
            raise ValueError("fan_polygon needs n >= 3")
        tris = tuple((0, i, i + 1) for i in range(1, n - 1))
        # triangle index i-1 hosts T_i = (0, i, i+1):
        #   a = (0, i), b = (i, i+1), c = (i+1, 0)
        gluings = tuple(
            (((i - 1) - 1, ROLE_C), (i - 1, ROLE_A))   # (0,i): T_{i-1}.c = T_i.a
            for i in range(2, n - 1)
        )
        boundary = [(0, ROLE_A)]
        boundary += [(i - 1, ROLE_B) for i in range(1, n - 1)]
        boundary += [(n - 3, ROLE_C)]
        return cls(tris, gluings, tuple(boundary))

    def side_of_polygon(self, j: int):
        """Boundary slot of polygon side j (fan convention)."""
        return self.boundary_sides[j]
