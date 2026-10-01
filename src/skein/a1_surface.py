"""Triangulated surfaces and their BPS quivers (FST map).

Migrated from `the archived tree`  on 2026-05-15 per
the design record (A_1 migration).  Rename + minimal-adaptation pass; semantics
unchanged.

Combinatorial 2-complex representation of ideal triangulations of
bordered surfaces, with the Fomin-Shapiro-Thurston exchange-matrix
rule

    for each triangle T with cyclic edge order (e_1, e_2, e_3):
        for k = 1, 2, 3 :  let  i = e_k ,  j = e_{k+1 mod 3} .
        if both  i  and  j  are internal :
            B[i, j]  +=  1          (and B[j, i]  -=  1  by antisymmetry).

Hand-coded constructors cover the atoms we need for the A_1 genus-0
sausage-chain dictionary:

* :func:`Triangulation.pentagon_disk`       --  disk, 5 marks (pentagon)
* :func:`Triangulation.pure_SU2_annulus`    --  annulus, (1, 1) marks
* :func:`Triangulation.disk_2marks_1puncture`  --  calibration case
                                                  (2 disconnected nodes)
* :func:`Triangulation.su2_nf2_disk`        --  SU(2) N_f=2 chamber
                                                  (disk, 1 mark + 2 punctures)
* :func:`Triangulation.pure_SU2_chain`      --  pure SU(2)^n linear chain
                                                  (annulus, 1+1 marks,
                                                   n-1 interior punctures)

Surface-level canonical form is deferred.  For now we match quivers
via :func:`bps_quiver_dictionary.signature` .
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Vertex:
    kind: str  # "puncture" or "marked"
    boundary: int | None = None


@dataclass(frozen=True)
class Edge:
    v1: int
    v2: int
    internal: bool

    @property
    def is_self_loop(self) -> bool:
        return self.v1 == self.v2


@dataclass(frozen=True)
class Triangle:
    edges: tuple[int, int, int]   # CCW cyclic order of edge indices


@dataclass
class Triangulation:
    vertices: list[Vertex]
    edges: list[Edge]
    triangles: list[Triangle]
    # Puncture decoration  ε  (WKB-paper §3.5 "tagged triangulations"):
    # a map  puncture_vertex_index -> ±1  carrying the choice of
    # monodromy-eigenline at each regular puncture.  Absent entries
    # default to  +1 .  The decoration is combinatorially invisible
    # (it does not affect the FST exchange matrix or any flip /
    # cut operation here), but it keys distinct tagged charts in
    # the  surface_catalog  and is toggled by :meth:`pop`.
    decoration: dict[int, int] = field(default_factory=dict)

    # --- derived quantities --------------------------------------

    @property
    def internal_edge_indices(self) -> list[int]:
        return [i for i, e in enumerate(self.edges) if e.internal]

    @property
    def n_internal(self) -> int:
        return sum(1 for e in self.edges if e.internal)

    # --- tagged-triangulation helpers ----------------------------

    def punctures(self) -> list[int]:
        """Indices of puncture vertices."""
        return [i for i, v in enumerate(self.vertices) if v.kind == "puncture"]

    def decoration_tuple(self) -> tuple[tuple[int, int], ...]:
        """Canonical tuple form of the decoration:  sorted
        ``((puncture_idx, sign), ...)``  for puncture vertices whose
        sign is  -1 .  ``+1`` -signed punctures (the default) are
        omitted, so an undecorated triangulation has the empty tuple
        as its decoration.  Used for hashing / catalog keying."""
        return tuple(sorted(
            (p, -1) for p in self.punctures() if self.decoration.get(p, 1) == -1
        ))

    def decoration_at(self, puncture: int) -> int:
        """Sign  ±1  at a puncture vertex (default  +1 )."""
        if puncture < 0 or puncture >= len(self.vertices):
            raise ValueError(f"puncture index {puncture} out of range")
        if self.vertices[puncture].kind != "puncture":
            raise ValueError(
                f"vertex {puncture} is {self.vertices[puncture].kind!r}, "
                f"not a puncture"
            )
        return self.decoration.get(puncture, 1)

    def pop(self, puncture: int) -> "Triangulation":
        """Pop a puncture:  ε[P] → -ε[P] .  Combinatorially trivial
        (same  vertices / edges / triangles ); only the decoration
        changes.  Mirrors the abstract pop morphism  π_P  in the
        WKB-paper §3.5 tagged-triangulation groupoid (decorated
        triangulations have a unique morphism between them given
        by a sequence of flips and pops; pops commute with
        everything and satisfy  π_P^2 = id )."""
        if puncture < 0 or puncture >= len(self.vertices):
            raise ValueError(f"puncture index {puncture} out of range")
        if self.vertices[puncture].kind != "puncture":
            raise ValueError(
                f"vertex {puncture} is {self.vertices[puncture].kind!r}, "
                f"not a puncture"
            )
        new_dec = dict(self.decoration)
        new_sign = -self.decoration.get(puncture, 1)
        if new_sign == 1:
            new_dec.pop(puncture, None)
        else:
            new_dec[puncture] = new_sign
        return Triangulation(
            vertices=list(self.vertices),
            edges=list(self.edges),
            triangles=list(self.triangles),
            decoration=new_dec,
        )

    def vertex_valence(self, v: int) -> int:
        """Number of edge-incidences on vertex  v  (self-loops count 2)."""
        count = 0
        for e in self.edges:
            if e.v1 == v:
                count += 1
            if e.v2 == v:
                count += 1
        return count

    def flavour_charge(self, puncture: int) -> tuple[int, ...]:
        """Charge  γ_f(P) ∈ Γ  associated to puncture  P , expressed in
        the internal-edge basis (one coordinate per entry of
        ``internal_edge_indices`` ).  Concretely, the sum of edges
        incident to  P , with self-loops counted twice (matching the
        valence convention).

        For a **degenerate / valence-2** puncture, this charge lies
        in  ker B  (the flavour sublattice), and the pop at  P  is
        the "simple" self-folded-face pop of WKB-paper §3.5.1 --
        a tropical cluster mutation at the loop-edge node with no
        side effects elsewhere.  For punctures with richer
        neighbourhoods,  γ_f(P)  is not in  ker B  in general, and
        the pop reduces to a "flip to a degenerate triangulation,
        pop, flip back" composite (WKB eq 4553-4558).
        """
        internal = self.internal_edge_indices
        vec = [0] * len(internal)
        for node_idx, edge_idx in enumerate(internal):
            e = self.edges[edge_idx]
            if e.v1 == puncture:
                vec[node_idx] += 1
            if e.v2 == puncture:
                vec[node_idx] += 1
        return tuple(vec)

    def drop_lonely_triangles(self) -> "Triangulation":
        """Return a new triangulation with all "lonely" triangles
        removed, where a lonely triangle is one whose three edges are
        all boundary (i.e., the triangle shares no edge with any
        other).  Vertices and edges that become unused are also
        pruned so the returned triangulation is compact.

        A no-op if no lonely triangles are present.  Does not change
        the FST exchange matrix (lonely triangles contribute no
        internal-edge incidences)."""
        internal = set(self.internal_edge_indices)

        def has_internal(T: Triangle) -> bool:
            return any(e in internal for e in T.edges)

        keep_tris = [T for T in self.triangles if has_internal(T)]
        if len(keep_tris) == len(self.triangles):
            return self
        # Rebuild edge / vertex lists, dropping those used only by
        # the dropped triangles.
        used_edges: set[int] = set()
        for T in keep_tris:
            used_edges.update(T.edges)
        old_to_new_e: dict[int, int] = {}
        new_edges: list[Edge] = []
        for old_idx, e in enumerate(self.edges):
            if old_idx in used_edges:
                old_to_new_e[old_idx] = len(new_edges)
                new_edges.append(e)
        used_vertices: set[int] = set()
        for e in new_edges:
            used_vertices.add(e.v1)
            used_vertices.add(e.v2)
        old_to_new_v: dict[int, int] = {}
        new_vertices: list[Vertex] = []
        for old_idx, v in enumerate(self.vertices):
            if old_idx in used_vertices:
                old_to_new_v[old_idx] = len(new_vertices)
                new_vertices.append(v)
        new_edges = [
            Edge(old_to_new_v[e.v1], old_to_new_v[e.v2], internal=e.internal)
            for e in new_edges
        ]
        new_tris = [
            Triangle(edges=tuple(old_to_new_e[oe] for oe in T.edges))
            for T in keep_tris
        ]
        # Carry the decoration over, dropping entries for vertices that
        # no longer exist.
        new_dec = {
            old_to_new_v[k]: v
            for k, v in self.decoration.items()
            if k in old_to_new_v
        }
        return Triangulation(
            vertices=new_vertices,
            edges=new_edges,
            triangles=new_tris,
            decoration=new_dec,
        )

    def is_connected(self) -> bool:
        """True iff the triangulation is connected via shared
        internal edges (triangles are nodes, internal edges are
        adjacencies).  Lonely triangles (no internal edges) count as
        isolated singleton components and therefore make the
        triangulation disconnected if any sibling triangles exist.

        An empty triangulation is vacuously connected."""
        if not self.triangles:
            return True
        internal = set(self.internal_edge_indices)
        edge_to_tris: dict[int, list[int]] = {}
        for ti, T in enumerate(self.triangles):
            for e in T.edges:
                if e in internal:
                    edge_to_tris.setdefault(e, []).append(ti)
        visited = {0}
        stack = [0]
        while stack:
            cur = stack.pop()
            for e in self.triangles[cur].edges:
                if e in internal:
                    for other in edge_to_tris.get(e, []):
                        if other not in visited:
                            visited.add(other)
                            stack.append(other)
        return len(visited) == len(self.triangles)

    def bps_quiver(self) -> list[list[int]]:
        """Exchange matrix  B  on internal edges via the FST rule."""
        internal = self.internal_edge_indices
        pos = {idx: k for k, idx in enumerate(internal)}
        n = len(internal)
        B = [[0] * n for _ in range(n)]
        for T in self.triangles:
            for k in range(3):
                i_edge = T.edges[k]
                j_edge = T.edges[(k + 1) % 3]
                if i_edge in pos and j_edge in pos:
                    i, j = pos[i_edge], pos[j_edge]
                    B[i][j] += 1
                    B[j][i] -= 1
        return B

    def flip(self, edge_index: int) -> "Triangulation":
        """FST quadrilateral retriangulation ("edge flip").

        For an internal edge  e = (u, v)  shared by two triangles  T1, T2
        with third corners  x  (of T1) and  y  (of T2), remove  e  and
        insert the new diagonal  (x, y) ; the two triangles are
        replaced by  (u, x, y)  and  (v, x, y) .

        Corresponds at the BPS-quiver level to mutation at the node
        associated with  ``edge_index`` : the FST quiver of the
        flipped triangulation equals the cluster-algebra mutation of
        the original.

        Raises:
          - ``ValueError``  if the edge is boundary.
          - ``NotImplementedError``  for self-loops or degenerate
            quadrilaterals (T1 and T2 share their third corner);
            these need special tagged-triangulation handling.
        """
        if not (0 <= edge_index < len(self.edges)):
            raise ValueError(f"edge_index {edge_index} out of range")
        e = self.edges[edge_index]
        if not e.internal:
            raise ValueError(f"edge {edge_index} is not internal")
        # Self-loops: flip produces a valid triangulation (self-loop
        # at vertex v replaced by a direct edge x-y between the third
        # corners of the two adjacent triangles), but the FST-derived
        # B of the flipped tri does NOT equal the single-node cluster
        # mutation of the original B -- it equals a DOUBLE mutation
        # (at the self-loop node plus at an adjacent node).  This
        # reflects the "self-loop = double node" phenomenon at the
        # BPS quiver level.  Callers that need FST == single mutation
        # should catch this case separately.
        u, v = e.v1, e.v2
        adj = [i for i, T in enumerate(self.triangles) if edge_index in T.edges]
        if len(adj) != 2:
            raise ValueError(
                f"edge {edge_index} appears in {len(adj)} triangles; "
                f"flip expects exactly 2"
            )
        T1_idx, T2_idx = adj
        T1 = self.triangles[T1_idx]
        T2 = self.triangles[T2_idx]

        def _third_corner(T):
            others = [ei for ei in T.edges if ei != edge_index]
            vsets = [{self.edges[o].v1, self.edges[o].v2} for o in others]
            if not vsets:
                return None
            shared = set.intersection(*vsets)
            shared -= {u, v}
            return next(iter(shared)) if len(shared) == 1 else None

        x = _third_corner(T1)
        y = _third_corner(T2)
        if x is None or y is None:
            # Parallel-arc bigon case: the "third corner" of an
            # adjacent triangle lies in  {u, v}  (the endpoints of the
            # flipped edge).  Geometrically, the edge has a parallel
            # companion arc (both between  u  and  v  in different
            # homotopy classes, e.g. the two arcs of  pure SU(2)
            # annulus winding oppositely around the cylinder).  At the
            # combinatorial level we handle this by reversing the
            # cyclic edge-order of both adjacent triangles; the FST
            # rule then produces the cluster-algebra mutation of the
            # BPS quiver at this node.
            def _is_parallel_bigon_triangle(T):
                for ei in T.edges:
                    if ei == edge_index:
                        continue
                    e_other = self.edges[ei]
                    if (e_other.v1 not in {u, v}
                            or e_other.v2 not in {u, v}):
                        return False
                return True
            if (_is_parallel_bigon_triangle(T1)
                    and _is_parallel_bigon_triangle(T2)):
                new_T1 = Triangle(edges=tuple(reversed(T1.edges)))
                new_T2 = Triangle(edges=tuple(reversed(T2.edges)))
                new_triangles = list(self.triangles)
                new_triangles[T1_idx] = new_T1
                new_triangles[T2_idx] = new_T2
                return Triangulation(
                    vertices=list(self.vertices),
                    edges=list(self.edges),
                    triangles=new_triangles,
                    decoration=dict(self.decoration),
                )
            raise NotImplementedError(
                "flip: could not identify a unique third corner "
                "(non-standard quadrilateral, not of the "
                "parallel-arc-bigon form)"
            )
        if x == y:
            raise NotImplementedError(
                "flip: degenerate quadrilateral (both adjacent triangles "
                "share the same third corner)"
            )

        # Distinguish the two non-e edges of each triangle by cyclic
        # position (next vs. previous relative to the flipped edge).
        # For non-self-loop flips this is equivalent to the
        # endpoint-matching convention (u = next-cyclic, v = previous-
        # cyclic); for self-loops where  u == v , cyclic position is
        # the only way to tell the two sides apart.
        def _edge_at_cyclic(T, direction: str) -> int:
            idx = list(T.edges).index(edge_index)
            if direction == "next":
                return T.edges[(idx + 1) % 3]
            return T.edges[(idx - 1) % 3]

        p1_next = _edge_at_cyclic(T1, "next")
        p1_prev = _edge_at_cyclic(T1, "prev")
        p2_next = _edge_at_cyclic(T2, "next")
        p2_prev = _edge_at_cyclic(T2, "prev")
        # For a self-loop flip, the two adjacent triangles have a
        # reversed orientation relationship (the "shared edge" is the
        # self-loop at a single vertex, so T2 effectively looks at
        # T1 from the "other side" of the loop).  Empirically,
        # swapping  p2_next  with  p2_prev  in the flip formulas
        # gives the correct single-node cluster mutation at FST level
        # for self-loop flips, while the non-swapped convention is
        # correct for non-self-loop flips.
        if e.is_self_loop:
            p2_next, p2_prev = p2_prev, p2_next
        # Replace the flipped edge with the new diagonal (x, y) , reusing
        # the same slot so other triangles' edge indices stay valid.
        new_edges = list(self.edges)
        new_edges[edge_index] = Edge(x, y, internal=True)

        # Build the new triangles.  Cyclic orders chosen so the FST
        # rule on the flipped triangulation equals cluster-algebra
        # mutation of the original at node  edge_index .
        new_T1 = Triangle(edges=(p1_next, edge_index, p2_next))
        new_T2 = Triangle(edges=(p2_prev, edge_index, p1_prev))

        new_triangles = list(self.triangles)
        new_triangles[T1_idx] = new_T1
        new_triangles[T2_idx] = new_T2

        return Triangulation(
            vertices=list(self.vertices),
            edges=new_edges,
            triangles=new_triangles,
            decoration=dict(self.decoration),
        )

    def tagged_flip(self, edge_index: int) -> "Triangulation":
        """Flip in the tagged-triangulation regime (WKB-paper §3.5.1).

        Same as :meth:`flip`  when the quadrilateral is
        non-degenerate.  On degenerate configurations this method
        handles two cases the untagged flip refuses:

        * **Degenerate quadrilateral** (both adjacent triangles
          share the same third corner  x , with  x ∉ {u, v} ):
          inserts a self-loop at  x  as the new diagonal -- the two
          resulting triangles form a self-folded face at  x  plus a
          partner radial-edge triangle (WKB fig `degenerate-face` ).

        * **Bigon + self-loop adjacency** (one adjacent triangle has
          shape  (E_1, E_2, loop)  with parallel edges  E_1, E_2
          between  (u, v)  and a self-loop at one of them, a
          configuration that appears around valence-2 punctures
          enclosed in a bigon of a sausage chain):  the pivot
          vertex of the degenerate triangle *is* one of  {u, v} ;
          we recover it by relaxing the "distinct third corner"
          restriction and proceed with the tagged flip using that
          pivot.

        Raises the same errors as :meth:`flip`  in other cases.
        """
        if not (0 <= edge_index < len(self.edges)):
            raise ValueError(f"edge_index {edge_index} out of range")
        e = self.edges[edge_index]
        if not e.internal:
            raise ValueError(f"edge {edge_index} is not internal")
        u, v = e.v1, e.v2
        adj = [i for i, T in enumerate(self.triangles) if edge_index in T.edges]
        if len(adj) != 2:
            raise ValueError(
                f"edge {edge_index} appears in {len(adj)} triangles; "
                f"tagged_flip expects exactly 2"
            )
        T1_idx, T2_idx = adj
        T1 = self.triangles[T1_idx]
        T2 = self.triangles[T2_idx]

        def _third_corner(T, *, relaxed: bool = False):
            """Pivot vertex of  T  opposite  edge_index .

            Strict (default): the unique vertex shared by the two
            non-flipped edges and NOT on the flipped edge.  Returns
            None if no such unique vertex exists.

            Relaxed: allow the shared vertex to also lie on the
            flipped edge (for bigon+self-loop adjacencies where the
            degenerate triangle has its pivot at  u  or  v ).
            """
            others = [ei for ei in T.edges if ei != edge_index]
            vsets = [{self.edges[o].v1, self.edges[o].v2} for o in others]
            if not vsets:
                return None
            shared = set.intersection(*vsets)
            if not relaxed:
                shared = shared - {u, v}
            return next(iter(shared)) if len(shared) == 1 else None

        strict_x = _third_corner(T1)
        strict_y = _third_corner(T2)
        x = strict_x if strict_x is not None else _third_corner(T1, relaxed=True)
        y = strict_y if strict_y is not None else _third_corner(T2, relaxed=True)

        if x is None or y is None:
            # Still couldn't pin down a pivot; defer to the untagged
            # flip (which will either succeed via the parallel-arc
            # bigon branch or raise).
            return self.flip(edge_index)

        both_strict = (strict_x is not None and strict_y is not None)
        if x != y and both_strict:
            # Standard non-degenerate case (strictly-distinct third
            # corners): the untagged flip handles it correctly.
            return self.flip(edge_index)

        # At least one of the adjacent triangles has a degenerate
        # (bigon + self-loop, or "both share pivot") structure -- we
        # use the generalised flip construction below: new diagonal
        # goes from  x  to  y  (a self-loop when x == y ), and the
        # cyclic-order recipe constructs two replacement triangles.
        # Reuse the  edge_index  slot so other triangles' edge
        # references stay valid.
        def _edge_at_cyclic(T, direction: str) -> int:
            idx = list(T.edges).index(edge_index)
            if direction == "next":
                return T.edges[(idx + 1) % 3]
            return T.edges[(idx - 1) % 3]

        p1_next = _edge_at_cyclic(T1, "next")
        p1_prev = _edge_at_cyclic(T1, "prev")
        p2_next = _edge_at_cyclic(T2, "next")
        p2_prev = _edge_at_cyclic(T2, "prev")

        new_edges = list(self.edges)
        new_edges[edge_index] = Edge(x, y, internal=True)

        new_T1 = Triangle(edges=(p1_next, edge_index, p2_next))
        new_T2 = Triangle(edges=(p2_prev, edge_index, p1_prev))

        new_triangles = list(self.triangles)
        new_triangles[T1_idx] = new_T1
        new_triangles[T2_idx] = new_T2

        return Triangulation(
            vertices=list(self.vertices),
            edges=new_edges,
            triangles=new_triangles,
            decoration=dict(self.decoration),
        )

    def cut_edge(self, edge_index: int) -> "Triangulation":
        """Cut the triangulation along an internal edge.

        Effect on the combinatorial 2-complex:

            - the internal edge at ``edge_index``  becomes a boundary
              edge;
            - a new boundary edge with the same endpoints is appended
              (the "other side" of the cut);
            - the two puncture endpoints (milestone-1 scope) are
              promoted to marked points on a NEW boundary component;
            - one of the two triangles that referenced the cut edge is
              updated to reference the appended boundary edge so each
              boundary copy is in exactly one triangle.

        Milestone 1 scope: non-self-loop, both endpoints are punctures
        (puncture-puncture cut).  Does not yet handle self-loops,
        punctures-to-boundary-marks, or disconnection.

        Effect on the FST-derived BPS quiver: exactly the removal of
        the row/column for this edge -- matching the quiver-level
        ``bps_quiver_dictionary.freeze``  operation on the
        corresponding node.  That equivalence is the validation test
        for this function.
        """
        if not (0 <= edge_index < len(self.edges)):
            raise ValueError(f"edge_index {edge_index} out of range")
        e = self.edges[edge_index]
        if not e.internal:
            raise ValueError(f"edge {edge_index} is not internal")
        if e.is_self_loop:
            raise NotImplementedError(
                "cut_edge on self-loops not yet implemented"
            )
        v1, v2 = e.v1, e.v2
        kinds = (self.vertices[v1].kind, self.vertices[v2].kind)
        new_vertices = list(self.vertices)

        if kinds == ("puncture", "puncture"):
            # p-p cut (milestone 1): both endpoints promoted to marks
            # on a NEW boundary component.
            existing_boundaries = {
                v.boundary for v in self.vertices if v.boundary is not None
            }
            new_bdry = (max(existing_boundaries) + 1) if existing_boundaries else 0
            new_vertices[v1] = Vertex(kind="marked", boundary=new_bdry)
            new_vertices[v2] = Vertex(kind="marked", boundary=new_bdry)
        elif "puncture" in kinds and "marked" in kinds:
            # p-b cut (milestone 2): puncture joins the existing
            # boundary, the existing mark is duplicated.  Net: that
            # boundary's mark count grows by 2.
            if self.vertices[v1].kind == "puncture":
                pv, mv = v1, v2
            else:
                pv, mv = v2, v1
            bnd = self.vertices[mv].boundary
            # Promote puncture to marked on the same boundary.
            new_vertices[pv] = Vertex(kind="marked", boundary=bnd)
            # Duplicate the mark mv as a new vertex on the same boundary.
            # (The "two sides of the slit" share the same marked point
            # location; combinatorially we track it as two distinct
            # marked vertices. The fan-of-triangles around  mv  is not
            # split in detail here -- we only need the mark count and
            # the FST edge structure for the quiver match.)
            new_mv = len(new_vertices)
            new_vertices.append(Vertex(kind="marked", boundary=bnd))
            # Retain  mv, new_mv  for the edge-endpoint update below.
        elif kinds == ("marked", "marked"):
            # m-m cut: both endpoints are marks.  Two subcases:
            #  (a) same boundary (a, a+2)  -- disconnects a small
            #      triangle carrying the  a+1  marking; main component
            #      is returned, triangle discarded.
            #  (b) different boundaries    -- merges the two boundaries
            #      into one; the cut's endpoints each gain a duplicate
            #      mark (the "other side" of the slit) and a second
            #      boundary edge joins the duplicates.
            b1 = self.vertices[v1].boundary
            b2 = self.vertices[v2].boundary
            if b1 != b2:
                # Merge boundaries b1 and b2 into a new boundary label.
                existing_boundaries = {
                    v.boundary for v in self.vertices if v.boundary is not None
                }
                new_bnd = (max(existing_boundaries) + 1
                           if existing_boundaries else 0)
                new_vertices = list(self.vertices)
                for i, vv in enumerate(new_vertices):
                    if vv.boundary in (b1, b2):
                        new_vertices[i] = Vertex(kind="marked",
                                                  boundary=new_bnd)
                # Duplicate v1 and v2 (the "other side" of the slit).
                v1_copy = len(new_vertices)
                new_vertices.append(Vertex(kind="marked",
                                            boundary=new_bnd))
                v2_copy = len(new_vertices)
                new_vertices.append(Vertex(kind="marked",
                                            boundary=new_bnd))
                # Flip cut edge to boundary; append a second boundary
                # edge between the duplicated endpoints.
                new_edges = list(self.edges)
                new_edges[edge_index] = Edge(v1, v2, internal=False)
                new_second_edge = len(new_edges)
                new_edges.append(Edge(v1_copy, v2_copy, internal=False))
                # One of the two adjacent triangles is updated to
                # reference the appended boundary edge (and its
                # triangle-local endpoints are relabeled to the copies).
                adj_tri_indices = [
                    i for i, T in enumerate(self.triangles)
                    if edge_index in T.edges
                ]
                if len(adj_tri_indices) != 2:
                    raise ValueError(
                        f"edge {edge_index} appears in "
                        f"{len(adj_tri_indices)} triangles; expected 2"
                    )
                new_triangles = list(self.triangles)
                t_to_update = adj_tri_indices[1]
                T = new_triangles[t_to_update]
                new_T_edges = tuple(
                    new_second_edge if ei == edge_index else ei
                    for ei in T.edges
                )
                new_triangles[t_to_update] = Triangle(edges=new_T_edges)
                # Within  t_to_update , also rewrite the non-cut edges'
                # endpoints from (v1,v2) to (v1_copy,v2_copy) so each
                # boundary copy is used by exactly one triangle side.
                # Each non-cut edge touching v1 or v2 in T has its
                # endpoint replaced by the copy IF it also lies on
                # t_to_update .  Simple approach: for each edge of
                # t_to_update other than the appended one, rewrite v1 ->
                # v1_copy and v2 -> v2_copy in its endpoints.
                for ei in new_T_edges:
                    if ei == new_second_edge:
                        continue
                    e_obj = new_edges[ei]
                    new_v1 = v1_copy if e_obj.v1 == v1 else (
                        v2_copy if e_obj.v1 == v2 else e_obj.v1
                    )
                    new_v2 = v1_copy if e_obj.v2 == v1 else (
                        v2_copy if e_obj.v2 == v2 else e_obj.v2
                    )
                    if (new_v1, new_v2) != (e_obj.v1, e_obj.v2):
                        new_edges[ei] = Edge(new_v1, new_v2,
                                              internal=e_obj.internal)
                return Triangulation(
                    vertices=new_vertices,
                    edges=new_edges,
                    triangles=new_triangles,
                    decoration=dict(self.decoration),
                )
            # Identify which adjacent triangle is the "small triangle
            # side": it's the triangle whose other 2 edges are BOTH
            # boundary edges of the same boundary component as v1/v2.
            adj_tri_indices_probe = [
                i for i, T in enumerate(self.triangles)
                if edge_index in T.edges
            ]
            if len(adj_tri_indices_probe) != 2:
                raise ValueError(
                    f"edge {edge_index} appears in "
                    f"{len(adj_tri_indices_probe)} triangles; expected 2"
                )
            triangle_side = None
            for ti in adj_tri_indices_probe:
                T = self.triangles[ti]
                other = [eidx for eidx in T.edges if eidx != edge_index]
                is_triangle_piece = all(
                    (not self.edges[oi].internal)
                    and self.edges[oi].v1 != self.edges[oi].v2
                    and self.vertices[self.edges[oi].v1].boundary == b1
                    and self.vertices[self.edges[oi].v2].boundary == b1
                    for oi in other
                )
                if is_triangle_piece:
                    triangle_side = ti
                    break
            if triangle_side is None:
                raise NotImplementedError(
                    "m-m cut: couldn't identify the disconnected "
                    "triangle side (the cut might not be of the "
                    "(a, a+2)-on-same-boundary form)"
                )
            # Main-side triangle (the one to keep).
            main_side = [
                ti for ti in adj_tri_indices_probe if ti != triangle_side
            ][0]
            # Compute the list of triangles to KEEP: everything except
            # the disconnected triangle.
            keep_triangles = [
                T for ti, T in enumerate(self.triangles)
                if ti != triangle_side
            ]
            # The 2 boundary edges of the disconnected triangle, and
            # the mark vertex a+1 between them, also go away.
            disconnected_T = self.triangles[triangle_side]
            disconnected_boundary_edges = [
                eidx for eidx in disconnected_T.edges if eidx != edge_index
            ]
            # a+1 = the vertex SHARED by the 2 disconnected boundary
            # edges (and not in {v1, v2}).  Intersect the edges' vertex
            # sets, then drop the cut-edge endpoints.
            edge_vertex_sets = [
                {self.edges[oi].v1, self.edges[oi].v2}
                for oi in disconnected_boundary_edges
            ]
            shared_vertex_candidates = set.intersection(*edge_vertex_sets)
            shared_vertex_candidates -= {v1, v2}
            if len(shared_vertex_candidates) != 1:
                raise ValueError(
                    f"m-m cut: could not identify the  a+1  vertex; "
                    f"candidates={shared_vertex_candidates}"
                )
            a_plus_1 = next(iter(shared_vertex_candidates))

            # Build new edge list: replace the cut edge with a boundary
            # edge (the "main side" now has this as its external
            # boundary), and REMOVE the 2 disconnected boundary edges +
            # the cut edge's role as bridge.  We'll keep indices stable
            # for unchanged edges by placing "removed" marker edges as
            # self-referring dummies -- but simpler: rebuild the edge
            # list from scratch with a reindexing.
            removed_edges = set([edge_index] + disconnected_boundary_edges)
            old_to_new: dict[int, int] = {}
            new_edges: list[Edge] = []
            for old_idx, oe in enumerate(self.edges):
                if old_idx == edge_index:
                    old_to_new[old_idx] = len(new_edges)
                    new_edges.append(Edge(v1, v2, internal=False))
                elif old_idx in removed_edges:
                    continue
                else:
                    old_to_new[old_idx] = len(new_edges)
                    new_edges.append(oe)
            # Build new vertex list: drop vertex a+1, reindex others.
            old_v_to_new: dict[int, int] = {}
            new_vertices = []
            for old_v, V in enumerate(self.vertices):
                if old_v == a_plus_1:
                    continue
                old_v_to_new[old_v] = len(new_vertices)
                new_vertices.append(V)
            # Reindex edge endpoints.
            new_edges = [
                Edge(old_v_to_new[e.v1], old_v_to_new[e.v2], internal=e.internal)
                for e in new_edges
            ]
            # Reindex triangles to use new edge indices.
            new_triangles = [
                Triangle(edges=tuple(old_to_new[oe] for oe in T.edges))
                for T in keep_triangles
            ]
            # Carry the decoration over, dropping dropped vertices.
            new_dec = {
                old_v_to_new[k]: v
                for k, v in self.decoration.items()
                if k in old_v_to_new
            }
            return Triangulation(
                vertices=new_vertices,
                edges=new_edges,
                triangles=new_triangles,
                decoration=new_dec,
            )
        else:
            raise NotImplementedError(
                f"cut_edge for endpoint kinds {kinds} not implemented"
            )

        # Find the 2 triangles referencing the cut edge.
        adj_tri_indices = [
            i for i, T in enumerate(self.triangles) if edge_index in T.edges
        ]
        if len(adj_tri_indices) != 2:
            raise ValueError(
                f"edge {edge_index} appears in {len(adj_tri_indices)} "
                f"triangles; cut_edge expects exactly 2 for a standard "
                f"internal edge"
            )

        # New edges: flip the cut edge to boundary, append a second
        # boundary arc with the same endpoints.
        new_edges = list(self.edges)
        new_edges[edge_index] = Edge(v1, v2, internal=False)
        new_second_edge = len(new_edges)
        new_edges.append(Edge(v1, v2, internal=False))

        # Update one of the two adjacent triangles to reference the
        # appended edge in place of the cut edge -- pick the second
        # occurrence so each boundary edge lives in exactly one triangle.
        new_triangles = list(self.triangles)
        t_to_update = adj_tri_indices[1]
        T = new_triangles[t_to_update]
        new_T_edges = tuple(
            new_second_edge if idx == edge_index else idx for idx in T.edges
        )
        new_triangles[t_to_update] = Triangle(edges=new_T_edges)

        # Cut promotes some former punctures (v1, v2 or one of them) to
        # marked vertices; drop their decoration entries.  The vertex
        # indices are unchanged -- only the kind changes.
        new_dec = {
            k: v for k, v in self.decoration.items()
            if new_vertices[k].kind == "puncture"
        }
        return Triangulation(
            vertices=new_vertices,
            edges=new_edges,
            triangles=new_triangles,
            decoration=new_dec,
        )

    def surface_topology(self) -> dict:
        """Coarse invariant:  (genus, n_punctures, boundary_marks) ."""
        V = len(self.vertices)
        E = len(self.edges)
        F = len(self.triangles)
        chi = V - E + F
        n_punctures = sum(1 for v in self.vertices if v.kind == "puncture")
        marks_per_component: dict[int, int] = {}
        for v in self.vertices:
            if v.kind == "marked":
                marks_per_component[v.boundary] = (
                    marks_per_component.get(v.boundary, 0) + 1
                )
        b = len(marks_per_component)
        genus = (2 - b - n_punctures - chi) // 2
        if genus < 0:
            genus = 0
        return {
            "genus": genus,
            "n_punctures": n_punctures,
            "boundary_marks": tuple(sorted(marks_per_component.values())),
            "euler": chi,
        }

    # --- hand-built atomic constructors --------------------------

    @classmethod
    def square_disk(cls) -> "Triangulation":
        """Square = disk with 4 marked boundary points.  Single
        internal diagonal, 2 triangles, trivial 1-node FST quiver
        ( B = [[0]] )."""
        vertices = [Vertex("marked", boundary=0) for _ in range(4)]
        edges = [
            Edge(0, 1, internal=False),
            Edge(1, 2, internal=False),
            Edge(2, 3, internal=False),
            Edge(3, 0, internal=False),
            Edge(0, 2, internal=True),     # diagonal
        ]
        triangles = [
            Triangle((0, 1, 4)),
            Triangle((2, 3, 4)),
        ]
        return cls(vertices=vertices, edges=edges, triangles=triangles)

    @classmethod
    def pentagon_disk(cls) -> "Triangulation":
        """Pentagon = disk with 5 marked boundary points."""
        vertices = [Vertex("marked", boundary=0) for _ in range(5)]
        edges = [
            Edge(0, 1, internal=False),
            Edge(1, 2, internal=False),
            Edge(2, 3, internal=False),
            Edge(3, 4, internal=False),
            Edge(4, 0, internal=False),
            Edge(0, 2, internal=True),    # d0
            Edge(0, 3, internal=True),    # d1
        ]
        B0, B1, B2, B3, B4, D0, D1 = 0, 1, 2, 3, 4, 5, 6
        triangles = [
            Triangle((B0, B1, D0)),
            Triangle((D0, B2, D1)),
            Triangle((D1, B3, B4)),
        ]
        return cls(vertices=vertices, edges=edges, triangles=triangles)

    @classmethod
    def pure_SU2_annulus(cls) -> "Triangulation":
        """Pure SU(2) = annulus with 1 marked point on each boundary.
        Two internal arcs wrap the cylinder in opposite homotopy
        classes; both triangles contain both, giving Kronecker-2."""
        vertices = [
            Vertex("marked", boundary=0),
            Vertex("marked", boundary=1),
        ]
        S0, S1, E0, E1 = 0, 1, 2, 3
        edges = [
            Edge(0, 0, internal=False),
            Edge(1, 1, internal=False),
            Edge(0, 1, internal=True),
            Edge(0, 1, internal=True),
        ]
        triangles = [
            Triangle((S0, E1, E0)),
            Triangle((E0, S1, E1)),
        ]
        return cls(vertices=vertices, edges=edges, triangles=triangles)

    @classmethod
    def disk_2marks_1puncture(cls) -> "Triangulation":
        """Calibration: disk with 2 boundary marks + 1 interior
        puncture.  Two internal edges (m1-p, p-m2) share both
        triangles with opposite cyclic order -> net zero pairing =
        two disconnected nodes in the BPS quiver."""
        vertices = [
            Vertex("marked", boundary=0),   # 0 : m1
            Vertex("marked", boundary=0),   # 1 : m2
            Vertex("puncture"),             # 2 : p
        ]
        M1, M2, P = 0, 1, 2
        B1 = 0
        B2 = 1
        E_M1P = 2
        E_M2P = 3
        edges = [
            Edge(M1, M2, internal=False),
            Edge(M1, M2, internal=False),
            Edge(M1, P, internal=True),
            Edge(M2, P, internal=True),
        ]
        triangles = [
            Triangle((E_M1P, E_M2P, B1)),
            Triangle((B2, E_M2P, E_M1P)),
        ]
        return cls(vertices=vertices, edges=edges, triangles=triangles)

    @classmethod
    def su2_nf2_disk(cls) -> "Triangulation":
        """SU(2) N_f=2 in the geometrically transparent chamber.
        Disk with 1 marked boundary point M + 2 punctures P1, P2.
        Internal edges: P2-M arc 1, P2-M arc 2, P1-P2, P1-M.
        Three triangles.  BPS quiver arrows  1->2, 1->4, 2->3,
        3->1, 4->2 ; edges 3 and 4 cancel (opposite orientation)."""
        vertices = [
            Vertex("marked", boundary=0),   # 0 : M
            Vertex("puncture"),             # 1 : P1
            Vertex("puncture"),             # 2 : P2
        ]
        M, P1, P2 = 0, 1, 2
        OUTER = 0
        E1, E2, E3, E4 = 1, 2, 3, 4
        edges = [
            Edge(M, M, internal=False),
            Edge(P2, M, internal=True),
            Edge(P2, M, internal=True),
            Edge(P1, P2, internal=True),
            Edge(P1, M, internal=True),
        ]
        triangles = [
            Triangle((E3, E1, E4)),
            Triangle((E4, E2, E3)),
            Triangle((E1, E2, OUTER)),
        ]
        return cls(vertices=vertices, edges=edges, triangles=triangles)

    @classmethod
    def SU2_chain_flavoured(cls, n: int,
                            left_nf: int = 0,
                            right_nf: int = 0) -> "Triangulation":
        """Sausage chain with 0, 1, or 2 single-arrow flavour caps
        at either end.

        Supported: ``left_nf, right_nf  in  {0, 1, 2}`` .

        Cap recipes:

        **1 flavour at the left end.**  The old  m1-m1  boundary
        self-loop becomes internal (= a fundamental-matter node);
        a new marked point  m1'  is added to the same boundary;
        two new boundary arcs  m1-m1'  are added; one new triangle
         (old loop,  m1-m1' arc 1, m1-m1' arc 2)  is attached.

        **2 flavours at the left end** (the author's mutated-flavour
        recipe -- matches the "mutation at a flavour node of the
        standard SU(2)^n quiver with 2 flavours at one end").  We
        drop  m1  entirely; the left-end boundary has two marked
        points  0  and  0' .  Let  P1  be the first interior
        puncture (new for  n = 1 , where it replaces the single
        gauge chain segment  m1->m2  with  0->P1  and  P1->m2 --
        so the chain logically lengthens by 1 at this end).
        Edges introduced:

            - ``1-1``  self-loop at  P1  (bifundamental)
            - two arcs  ``1-0``  (first SU(2) gauge pair, pairing 1
              not 2 because only one triangle contains both)
            - boundary arc  ``0-0'``  (standard fundamental)
            - internal edge  ``1-0'``  (mutated fundamental)
            - boundary arc  ``0'-0`` going the other way around

        Triangles:

            - ``1-0-1`` :  (1-0 arc a, 1-0 arc b, 1-1 loop)
            - ``1-0-0'`` :  (1-0 arc a, 0-0' boundary, 1-0')
            - ``1-0'-0`` :  (1-0', 0'-0 boundary, 1-0 arc b)

        The  0-0'  boundary arc and the original ``0'``-to-``0`` arc
        close the cap's boundary.

        Right end mirrors the left end in the obvious way.

        Implementation strategy: start from ``pure_SU2_chain(n)`` ,
        then apply cap transformations at each end.  For  nf=1  we
        reuse the existing single-flavour recipe.  For  nf=2  we
        rebuild the end structure from scratch.

        NOTE on  n=1 with  left_nf=2 : the "first interior puncture"
        for the cap is effectively injected at the left end, so the
        chain still has one gauge segment (0  to  m2) but with the
        cap's puncture P1 on the left side between them.  The chain
        logically becomes an n=2 structure on the left.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if left_nf not in (0, 1, 2) or right_nf not in (0, 1, 2):
            raise ValueError(
                f"left_nf, right_nf must be in {{0, 1, 2}} ; "
                f"got left_nf={left_nf}, right_nf={right_nf}"
            )
        # The 2-flavour cap at a given end reuses the chain's existing
        # adjacent puncture as the cap's "P1" (the author's recipe).
        # For n = 1  there is no such puncture, so 2-flavour caps
        # require  n >= 2 .
        if left_nf == 2 and n < 2:
            raise ValueError(
                "left_nf=2 requires n >= 2 (cap reuses the chain's "
                "first interior puncture)"
            )
        if right_nf == 2 and n < 2:
            raise ValueError(
                "right_nf=2 requires n >= 2 (cap reuses the chain's "
                "last interior puncture)"
            )

        # The effective chain length equals  n  (no extras); the
        # 2-flavour caps reuse existing punctures.
        n_eff = n

        # --- vertices ------------------------------------------------
        # Left-end marks:
        #   left_nf == 0  :  single mark  m1  at vertex 0
        #   left_nf == 1  :  two marks  m1, m1'  both on boundary 0
        #   left_nf == 2  :  two marks  0, 0'  both on boundary 0
        # Right-end symmetric.
        vertices: list[Vertex] = []

        # Left-end vertices.
        #   nf == 0 : single mark on boundary 0 (m1)
        #   nf == 1 : two marks on boundary 0 (m1, m1')
        #   nf == 2 : two punctures (0, 0'), no boundary left there
        #             (the Nf=2 cap closes off the boundary entirely)
        if left_nf == 0:
            vertices.append(Vertex("marked", boundary=0))
        elif left_nf == 1:
            vertices.append(Vertex("marked", boundary=0))
            vertices.append(Vertex("marked", boundary=0))
        else:   # left_nf == 2
            vertices.append(Vertex("puncture"))
            vertices.append(Vertex("puncture"))

        # Interior punctures of the effective chain.
        puncture_base = len(vertices)
        for _ in range(n_eff - 1):
            vertices.append(Vertex("puncture"))
        puncture_end = len(vertices)   # exclusive

        # Right-end vertices (mirror of left).
        # For right_nf == 0 / 1, the right boundary component index
        # is  1 .  For right_nf == 2 it's absent.
        if right_nf == 0:
            vertices.append(Vertex("marked", boundary=1))
            right_primary = len(vertices) - 1
            right_secondary = None
        elif right_nf == 1:
            vertices.append(Vertex("marked", boundary=1))
            right_primary = len(vertices) - 1
            vertices.append(Vertex("marked", boundary=1))
            right_secondary = len(vertices) - 1
        else:   # right_nf == 2
            vertices.append(Vertex("puncture"))
            right_primary = len(vertices) - 1
            vertices.append(Vertex("puncture"))
            right_secondary = len(vertices) - 1

        # Convenience vertex indices.
        LEFT_PRIMARY = 0                     # m1  or  0
        LEFT_SECONDARY = 1 if left_nf >= 1 else None   # m1' or 0'
        # Interior punctures are at indices  puncture_base ..
        # puncture_end - 1 .

        def puncture_idx(k: int) -> int:
            """k = 1 .. n_eff - 1 ,  returns vertex index."""
            return puncture_base + (k - 1)

        RIGHT_PRIMARY = right_primary
        RIGHT_SECONDARY = right_secondary

        # The "effective chain" has segments 0 .. n_eff - 1 connecting
        # the interior punctures.  Left endpoint is LEFT_PRIMARY (for
        # 0 or 1 flavour) or LEFT_PRIMARY=0 for 2 flavour (in which
        # case segment 0's left is still LEFT_PRIMARY=0 , connected to
        # puncture P1 = puncture_idx(1) ).
        def segment_left(k: int) -> int:
            if k == 0:
                return LEFT_PRIMARY
            return puncture_idx(k)

        def segment_right(k: int) -> int:
            if k == n_eff - 1:
                return RIGHT_PRIMARY
            return puncture_idx(k + 1)

        # --- edges ---------------------------------------------------
        # Layout:
        #   0  : left boundary self-loop  (LEFT_PRIMARY -- LEFT_PRIMARY)
        #        -- INTERNAL if left_nf == 1 (= flavour matter node)
        #        -- BOUNDARY if left_nf == 0
        #        -- NOT USED (removed) if left_nf == 2
        #   1  : right boundary self-loop (mirror)
        #   arcs (a_k, b_k)  for k = 0 .. n_eff - 1
        #   interior self-loops at effective-chain punctures
        #   extra edges/arcs added by each cap after the above
        # When left_nf == 2 we still emit edge index 0 but mark it
        # BOUNDARY with a dummy role -- this keeps the arc-indexing
        # stable -- and add explicit cap structure afterwards.  To
        # avoid a "dummy edge", we instead conditionally build the
        # edge list and remember the slot.
        edges: list[Edge] = []

        # Left-end self-loop (or absent for nf=2).
        if left_nf == 2:
            LEFT_SELFLOOP = None  # no such edge; the cap wires things
                                  # through its own pieces
        else:
            internal = (left_nf == 1)
            LEFT_SELFLOOP = len(edges)
            edges.append(Edge(LEFT_PRIMARY, LEFT_PRIMARY, internal=internal))

        # Right-end self-loop.
        if right_nf == 2:
            RIGHT_SELFLOOP = None
        else:
            internal = (right_nf == 1)
            RIGHT_SELFLOOP = len(edges)
            edges.append(Edge(RIGHT_PRIMARY, RIGHT_PRIMARY, internal=internal))

        # Gauge segment arcs for the effective chain.
        arc_a: dict[int, int] = {}
        arc_b: dict[int, int] = {}
        for k in range(n_eff):
            L = segment_left(k)
            R = segment_right(k)
            arc_a[k] = len(edges)
            edges.append(Edge(L, R, internal=True))
            arc_b[k] = len(edges)
            edges.append(Edge(L, R, internal=True))

        # Interior puncture self-loops.
        interior_loop: dict[int, int] = {}
        for k in range(1, n_eff):
            v = puncture_idx(k)
            interior_loop[k] = len(edges)
            edges.append(Edge(v, v, internal=True))

        # Cap-specific extras.
        # Left nf=1: two boundary arcs LEFT_PRIMARY <-> LEFT_SECONDARY
        # Left nf=2: mutated-flavour structure:
        #            boundary arc  LEFT_PRIMARY -- LEFT_SECONDARY
        #            boundary arc  back the other way
        #            internal edge  puncture_idx(1) -- LEFT_SECONDARY
        #            (the 1-0' mutated fundamental)
        left_bdry1 = left_bdry2 = None
        left_mutated_fund = None
        if left_nf == 1:
            left_bdry1 = len(edges)
            edges.append(Edge(LEFT_PRIMARY, LEFT_SECONDARY, internal=False))
            left_bdry2 = len(edges)
            edges.append(Edge(LEFT_SECONDARY, LEFT_PRIMARY, internal=False))
        elif left_nf == 2:
            # Ideal-triangulation cap: single internal 0-0' edge
            # (standard fundamental) + single internal 1-0' edge
            # (mutated fundamental).  No boundary arcs.
            left_std_fund = len(edges)
            edges.append(Edge(LEFT_PRIMARY, LEFT_SECONDARY, internal=True))
            left_mutated_fund = len(edges)
            edges.append(Edge(puncture_idx(1), LEFT_SECONDARY, internal=True))

        # Right-end symmetric.
        right_bdry1 = right_bdry2 = None
        right_mutated_fund = None
        if right_nf == 1:
            right_bdry1 = len(edges)
            edges.append(Edge(RIGHT_PRIMARY, RIGHT_SECONDARY, internal=False))
            right_bdry2 = len(edges)
            edges.append(Edge(RIGHT_SECONDARY, RIGHT_PRIMARY, internal=False))
        elif right_nf == 2:
            # Ideal-triangulation cap (mirror of left).
            right_std_fund = len(edges)
            edges.append(Edge(RIGHT_PRIMARY, RIGHT_SECONDARY, internal=True))
            right_mutated_fund = len(edges)
            edges.append(Edge(puncture_idx(n_eff - 1), RIGHT_SECONDARY,
                              internal=True))

        # --- triangles -----------------------------------------------
        def self_at_vertex(v: int, seg_k: int, is_left_side: bool) -> int | None:
            """Return the self-loop edge index incident to vertex v
            at the given side of segment seg_k.  Returns None when
            the 2-flavour cap has eaten the endpoint's self-loop --
            in that case the caller substitutes the cap's own
            structure."""
            if v == LEFT_PRIMARY:
                return LEFT_SELFLOOP   # may be None if left_nf == 2
            if v == RIGHT_PRIMARY:
                return RIGHT_SELFLOOP  # may be None if right_nf == 2
            # Interior puncture.
            for k, idx in interior_loop.items():
                if puncture_idx(k) == v:
                    return idx
            return None

        triangles: list[Triangle] = []

        # Main sausage pairs, with a twist at the 2-flavour ends:
        # the segment-0 left-side triangle uses the cap pieces
        # rather than a self-loop at LEFT_PRIMARY (same for right).
        for k in range(n_eff):
            L = segment_left(k)
            R = segment_right(k)
            a, b = arc_a[k], arc_b[k]

            # Left-side triangle  (L, R, L).
            if k == 0 and left_nf == 2:
                # Replaced by the three 2-flavour cap triangles below;
                # skip the usual left sausage triangle.
                pass
            else:
                s_L = self_at_vertex(L, k, is_left_side=True)
                triangles.append(Triangle((a, b, s_L)))

            # Right-side triangle  (R, L, R).
            if k == n_eff - 1 and right_nf == 2:
                pass
            else:
                s_R = self_at_vertex(R, k, is_left_side=False)
                triangles.append(Triangle((a, b, s_R)))

        # 1-flavour cap triangles: (old loop, new arc 1, new arc 2).
        if left_nf == 1:
            triangles.append(Triangle((LEFT_SELFLOOP, left_bdry1, left_bdry2)))
        if right_nf == 1:
            triangles.append(Triangle((RIGHT_SELFLOOP, right_bdry1, right_bdry2)))

        # 2-flavour cap triangles (the author's recipe, left end).  Note:
        # the  T_0^R  triangle already emitted in the main loop IS
        # the "1-0-1"  triangle (cyclic  (a0, b0, 1-1 loop) ), so we
        # don't duplicate it here.  Only the two "side" triangles
        # remain, and they use the internal  0-0'  and  1-0'  edges.
        if left_nf == 2:
            a0 = arc_a[0]
            b0 = arc_b[0]
            # 1-0-0' : (1-0 arc a, 0-0' std fund, 1-0' mutated fund)
            triangles.append(Triangle((a0, left_std_fund, left_mutated_fund)))
            # 1-0'-0 : (1-0' mutated fund, 0-0' std fund, 1-0 arc b)
            triangles.append(Triangle((left_mutated_fund, left_std_fund, b0)))

        # Right-end mirror.
        if right_nf == 2:
            last = n_eff - 1
            aL = arc_a[last]
            bL = arc_b[last]
            triangles.append(Triangle((aL, right_std_fund, right_mutated_fund)))
            triangles.append(Triangle((right_mutated_fund, right_std_fund, bL)))

        return cls(vertices=vertices, edges=edges, triangles=triangles)

    @classmethod
    def pure_SU2_chain(cls, n: int) -> "Triangulation":
        """Pure SU(2)^n linear chain (no fundamentals).

        Annulus with 2 marked boundary points m1, m2 and n-1 interior
        punctures  p_1, ..., p_{n-1} .  Edges:

          - outer boundary self-loops at m1 and m2
          - for each segment k = 0..n-1 : two parallel arcs between
            L_k = (m1 if k == 0 else p_k)  and
            R_k = (m2 if k == n-1 else p_{k+1})
          - for each interior puncture  p_k  (k = 1..n-1) : a self-loop
            p_k-p_k  (bifundamental matter node)

        Triangles: two per segment ("sausage" pair), both with cyclic
        edge order  (a_k, b_k, self-loop) -- left-based then right-
        based.  Under FST this gives

          * Kronecker-2 within each gauge pair  (a_k, b_k) ,
          * single-arrow bifund-to-gauge couplings around each
            interior self-loop.

        Expected quiver signature: matches
        LinearUQuiver([2]*n, [0]*n) .
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        vertices = [Vertex("marked", boundary=0)]
        for _ in range(1, n):
            vertices.append(Vertex("puncture"))
        vertices.append(Vertex("marked", boundary=1))

        BDRY_M1 = 0
        BDRY_M2 = 1

        def arc_a(k: int) -> int: return 2 + 2 * k
        def arc_b(k: int) -> int: return 2 + 2 * k + 1
        def interior_self(k: int) -> int: return 2 + 2 * n + (k - 1)
        def self_at_vertex(v: int) -> int:
            if v == 0:
                return BDRY_M1
            if v == n:
                return BDRY_M2
            return interior_self(v)

        edges = [
            Edge(0, 0, internal=False),   # BDRY_M1
            Edge(n, n, internal=False),   # BDRY_M2
        ]
        for k in range(n):
            L = 0 if k == 0 else k
            R = n if k == n - 1 else k + 1
            edges.append(Edge(L, R, internal=True))
            edges.append(Edge(L, R, internal=True))
        for k in range(1, n):
            edges.append(Edge(k, k, internal=True))

        triangles = []
        for k in range(n):
            L = 0 if k == 0 else k
            R = n if k == n - 1 else k + 1
            a, b = arc_a(k), arc_b(k)
            s_L = self_at_vertex(L)
            s_R = self_at_vertex(R)
            triangles.append(Triangle((a, b, s_L)))
            triangles.append(Triangle((a, b, s_R)))

        return cls(vertices=vertices, edges=edges, triangles=triangles)
