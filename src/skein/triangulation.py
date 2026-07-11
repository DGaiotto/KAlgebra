"""Ideal triangulations of closed punctured orientable surfaces.

Originally sphere-only; the constructor now accepts any genus (the
Euler check stores `genus`), and all the local machinery (sigma,
corners, arcs, state sums, resolutions) is genus-agnostic.

Two construction paths:

  * The strict Triangle-based constructor requires distinct-endpoint
    edges, at most one edge per vertex pair, and no self-folded
    triangles (n >= 3; every edge in exactly two distinct triangles;
    Euler relation).  Vertex tuples are authoritative there.
  * `from_edge_data` accepts self-loops and multi-edges (id-based
    edges).  It derives
    everything the engines need WITHOUT vertex tuples: per-slot
    orientation bits (`slot_forward` — does triangle ti traverse the
    edge at ccw slot pos along the edge's canonical direction?) with
    the two incidences of every edge traversing it oppositely (the
    orientable gluing completion of the underspecified edge data),
    and punctures as corner-walk orbits (validated against the
    nominal endpoint labels).  Self-folded
    triangles (a repeated edge index within one triangle) remain out
    of scope on BOTH paths: their two lifts live in the same triangle
    algebra and stop commuting (the `[y_e' y_e'']` Weyl bracket picks
    up a genuine q-power), and the matching machinery would need
    half-edge-addressed points.

The theory itself imposes none of these restrictions (two edges of
one triangle may be glued together); the self-folded gate is an
engine limitation, not mathematics.

Self-contained: no imports from elsewhere in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Edge:
    """Undirected ideal edge.

    Stored canonically as (u, v) with u <= v.  Self-loops (u == v) are
    permitted; for distinct endpoints, u < v.
    """
    u: int
    v: int

    def __post_init__(self):
        if self.u > self.v:
            raise ValueError(
                f"Edge requires u <= v (got u={self.u}, v={self.v})"
            )

    @classmethod
    def of(cls, a: int, b: int) -> "Edge":
        """Convenience constructor for the common (distinct-endpoints) case."""
        if a == b:
            raise ValueError(
                f"self-loop at puncture {a}: use Edge(a, a) directly "
                "(self-loops are supported by the relaxed `from_edge_data` "
                "constructor of Triangulation, but not by Edge.of)"
            )
        return cls(min(a, b), max(a, b))

    @property
    def is_self_loop(self) -> bool:
        return self.u == self.v

    def endpoints(self) -> frozenset[int]:
        return frozenset({self.u, self.v})


@dataclass(frozen=True)
class Triangle:
    """Ideal triangle as a ccw triple of puncture indices.

    The three edges in ccw order are
        (Edge.of(v0, v1), Edge.of(v1, v2), Edge.of(v2, v0)).
    The starting vertex of edge i is verts[i] and the ending vertex
    is verts[(i + 1) % 3].
    """
    verts: tuple[int, int, int]

    def __post_init__(self):
        if len(self.verts) != 3:
            raise ValueError(f"Triangle requires 3 vertices, got {self.verts}")
        if len(set(self.verts)) != 3:
            raise ValueError(
                f"Triangle requires 3 distinct vertices, got {self.verts}"
            )

    def edges_ccw(self) -> tuple[Edge, Edge, Edge]:
        v = self.verts
        return (
            Edge.of(v[0], v[1]),
            Edge.of(v[1], v[2]),
            Edge.of(v[2], v[0]),
        )

    def directed_edges_ccw(self) -> tuple[tuple[int, int], ...]:
        """The three boundary edges as ordered start->end tuples in ccw order."""
        v = self.verts
        return ((v[0], v[1]), (v[1], v[2]), (v[2], v[0]))


@dataclass
class Triangulation:
    """Ideal triangulation of S^2_{0,n}.

    Inputs:
        n_punctures   number of punctures (>= 3).
        triangles     list of Triangle objects in some fixed order.

    Derived (filled in __post_init__):
        edges                 canonical ordering of all edges
        edge_index            Edge -> position in `edges`
        triangle_edges        per triangle, the 3 edge indices in ccw order
        edge_incidences       per edge, list of (triangle_idx, position).
                              Always length 2 in this build.
    """
    n_punctures: int
    triangles: list[Triangle]

    edges: list[Edge] = field(init=False)
    genus: int = field(init=False)
    edge_index: dict[Edge, int] = field(init=False)
    triangle_edges: list[tuple[int, int, int]] = field(init=False)
    edge_incidences: list[list[tuple[int, int]]] = field(init=False)
    # Per (triangle, ccw slot): does the triangle traverse the edge at
    # that slot along the edge's canonical point-indexing direction?
    # Strict path: canonical = the stored u -> v direction.  Relaxed
    # path: canonical = the traversal direction of the edge's FIRST
    # incidence (the second traverses oppositely — orientable gluing).
    # This bit is the ONLY orientation datum the multicurve / Kauffman
    # engines need; nothing downstream reads vertex labels.
    slot_forward: list[tuple[bool, bool, bool]] = field(init=False)
    # Relaxed path only: punctures as corner-walk orbits, aligned so
    # that puncture_orbits[p] is the orbit of nominal puncture p.
    # None on the strict path (vertex tuples are authoritative there).
    puncture_orbits: list[list[tuple[int, int]]] | None = field(
        init=False, default=None,
    )
    # Bordered mode (from_edge_data(allow_boundary=True) only):
    # marked_points = the vertex labels lying on the boundary (each an
    # irregular-puncture marked point; its orbit is a corner CHAIN);
    # holes = the boundary circles, each a tuple of boundary edge ids
    # in boundary-walk cyclic order (an irregular puncture of rank
    # len(hole)/2; even length <=> a U(1) flavour).  Both empty on
    # closed charts.
    marked_points: tuple = field(init=False, default=())
    holes: tuple = field(init=False, default=())

    def __post_init__(self):
        if self.n_punctures < 3:
            raise ValueError(
                f"closed n-punctured sphere is triangulable only for n >= 3, "
                f"got n = {self.n_punctures}"
            )

        for t in self.triangles:
            for v in t.verts:
                if not (0 <= v < self.n_punctures):
                    raise ValueError(
                        f"triangle {t} has vertex {v} out of range "
                        f"[0, {self.n_punctures})"
                    )

        # collect edges in canonical (u, v) form
        edge_set: dict[Edge, None] = {}
        for t in self.triangles:
            for e in t.edges_ccw():
                edge_set.setdefault(e, None)
        edges = sorted(edge_set, key=lambda e: (e.u, e.v))
        edge_index = {e: i for i, e in enumerate(edges)}

        # per-triangle edge indices in ccw order
        triangle_edges: list[tuple[int, int, int]] = []
        for t in self.triangles:
            te = t.edges_ccw()
            triangle_edges.append((edge_index[te[0]], edge_index[te[1]], edge_index[te[2]]))

        # incidences
        edge_incidences: list[list[tuple[int, int]]] = [[] for _ in edges]
        for ti, te in enumerate(triangle_edges):
            for pos, ei in enumerate(te):
                edge_incidences[ei].append((ti, pos))

        # closed-sphere consistency: every edge in exactly two triangles
        for ei, inc in enumerate(edge_incidences):
            if len(inc) != 2:
                e = edges[ei]
                raise ValueError(
                    f"edge {ei} ({e.u}-{e.v}) appears in {len(inc)} triangles "
                    f"(expected 2 for closed sphere triangulation)"
                )
            t1, t2 = inc[0][0], inc[1][0]
            if t1 == t2:
                e = edges[ei]
                raise ValueError(
                    f"edge {ei} ({e.u}-{e.v}) is glued to itself in triangle {t1} "
                    "(self-folded triangles not supported in this build)"
                )

        # Euler relation cross-check
        n_e = len(edges)
        n_f = len(self.triangles)
        if 2 * n_e != 3 * n_f:
            raise ValueError(
                f"inconsistent triangulation: 2E = {2*n_e}, 3F = {3*n_f} (must be equal)"
            )
        chi = self.n_punctures - n_e + n_f
        if chi > 2 or chi % 2 != 0:
            raise ValueError(
                f"Euler characteristic V-E+F = {chi}: not of the form "
                "2 - 2g for a closed orientable surface"
            )
        self.genus = (2 - chi) // 2

        self.edges = edges
        self.edge_index = edge_index
        self.triangle_edges = triangle_edges
        self.edge_incidences = edge_incidences
        # orientation bits from the (authoritative) vertex tuples
        slot_forward: list[tuple[bool, bool, bool]] = []
        for t, te in zip(self.triangles, triangle_edges):
            bits = []
            for pos in range(3):
                e = edges[te[pos]]
                bits.append(t.verts[pos] == e.u)
            slot_forward.append(tuple(bits))
        self.slot_forward = slot_forward
        self.puncture_orbits = None

    # --- accessors ---------------------------------------------------

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_triangles(self) -> int:
        return len(self.triangle_edges)

    def edge_id(self, a: int, b: int) -> int:
        return self.edge_index[Edge.of(a, b)]

    def puncture_flavour_charge(self, puncture: int) -> tuple[int, ...]:
        """Lattice element f_p in ker(sigma) attached to puncture `p`.

        Defined as the sum (with multiplicity) of edges incident to p:
        each edge e = (u, v) contributes +1 to f_p[edge_index(e)] for
        each endpoint of e equal to p (so a self-loop at p contributes
        +2; a regular edge with one endpoint at p contributes +1).

        For closed triangulations (FST) the resulting f_p is always in
        ker(sigma) -- the 'flavour' direction associated to the
        puncture's holonomy.  On the relaxed `from_edge_data` path the
        charge is computed from the puncture's CORNER ORBIT (which is
        derived from the gluing itself), not from the nominal endpoint
        labels -- the labels only select which orbit is "puncture p",
        and their consistency with the gluing is validated at
        construction.
        """
        if not (0 <= puncture < self.n_punctures):
            raise ValueError(
                f"puncture {puncture} out of range [0, {self.n_punctures})"
            )
        if puncture in self.marked_points:
            raise ValueError(
                f"vertex {puncture} is a boundary MARKED POINT, not an "
                "interior puncture; marked points carry no flavour "
                "charge (the hole's U(1), if any, lives in the mutable "
                "block's kernel)"
            )
        if self.puncture_orbits is not None:
            f = [0] * self.n_edges
            for (ti, k) in self.puncture_orbits[puncture]:
                f[self.triangle_edges[ti][(k + 1) % 3]] += 1
            return tuple(f)
        f = [0] * self.n_edges
        for ei, e in enumerate(self.edges):
            if e.u == puncture:
                f[ei] += 1
            if e.v == puncture:
                f[ei] += 1
        return tuple(f)

    @staticmethod
    def _corner_orbits(
        triangle_edges: list[tuple[int, int, int]],
        edge_incidences: list[list[tuple[int, int]]],
    ) -> tuple[list[list[tuple[int, int]]], list[bool]]:
        """Vertices as orbits of corners under the walk around a vertex.

        Corner k of triangle ti sits at the vertex where the edge at ccw
        slot k+1 ends and the edge at ccw slot k+2 starts.  Crossing the
        outgoing edge (slot k+2) into its other incidence (tj, pj) lands
        on the corner of tj whose INCOMING slot is pj, i.e. k' = pj - 1.
        At a BOUNDARY edge (single incidence) the walk terminates, so
        the groups come in two kinds: CYCLES (interior vertices =
        regular punctures) and CHAINS (boundary vertices = the marked
        points of irregular punctures).  Returns `(orbits, is_chain)`.
        Uses only triangle_edges + edge_incidences -- no vertex labels.
        """
        step: dict[tuple[int, int], tuple[int, int] | None] = {}
        for ti, te in enumerate(triangle_edges):
            for k in range(3):
                pos_out = (k + 2) % 3
                inc = edge_incidences[te[pos_out]]
                if len(inc) == 1:
                    step[(ti, k)] = None       # boundary: walk terminates
                    continue
                (t1, p1), (t2, p2) = inc
                tj, pj = (t2, p2) if (t1, p1) == (ti, pos_out) else (t1, p1)
                step[(ti, k)] = (tj, (pj - 1) % 3)
        # chain starts: corners nobody steps into
        targets = {v for v in step.values() if v is not None}
        seen: set[tuple[int, int]] = set()
        orbits: list[list[tuple[int, int]]] = []
        is_chain: list[bool] = []
        for s in sorted(step):
            if s in seen or s in targets:
                continue
            ch, cur = [], s
            while cur is not None and cur not in seen:
                seen.add(cur)
                ch.append(cur)
                cur = step[cur]
            orbits.append(ch)
            is_chain.append(True)
        for c in sorted(step):
            if c in seen:
                continue
            orb, cur = [], c
            while cur not in seen:
                seen.add(cur)
                orb.append(cur)
                cur = step[cur]
            orbits.append(orb)
            is_chain.append(False)
        return orbits, is_chain

    def triangles_at_edge(self, ei: int) -> tuple[int, int]:
        inc = self.edge_incidences[ei]
        return (inc[0][0], inc[1][0])

    # --- bordered accessors (id-based bordered mode) -------

    @property
    def is_bordered(self) -> bool:
        return bool(self.holes)

    @property
    def boundary_edge_ids(self) -> list[int]:
        return [e for e, inc in enumerate(self.edge_incidences)
                if len(inc) == 1]

    @property
    def internal_edge_ids(self) -> list[int]:
        return [e for e, inc in enumerate(self.edge_incidences)
                if len(inc) == 2]

    def mutable_block(self) -> list[list[int]]:
        """The internal x internal principal block of `sigma()` — the BPS
        Dirac pairing of the canonical algebra on a bordered chart
        (boundary edges frozen; cf. `BorderedTriangulation.mutable_block`)."""
        s = self.sigma()
        ids = self.internal_edge_ids
        return [[s[i][j] for j in ids] for i in ids]

    def cut(self, edge: int) -> "Triangulation":
        """Un-glue internal `edge`: its two incidence slots become TWO
        boundary edges (the second gets a fresh id appended at the end),
        creating (part of) an irregular puncture — the static endpoint
        of the certified cut / node-drop RG.  Edge ids of all other
        edges are unchanged.

        Cutting an edge whose endpoint is ALREADY a marked point SPLITS
        that vertex (the boundary passes through it twice): the extra
        marked points get fresh labels (minted past the current
        n_punctures), everything else keeps its label.  Labels are
        re-derived from the corner orbits of the new gluing, so the
        result is A24-consistent by construction."""
        inc = self.edge_incidences[edge]
        if len(inc) != 2:
            raise ValueError(f"edge {edge} is already a boundary edge")
        (t1, p1), (t2, p2) = inc
        old = self.edges[edge]
        new_id = len(self.edges)
        new_tris = [list(te) for te in self.triangle_edges]
        new_tris[t2][p2] = new_id
        new_tris = [tuple(t) for t in new_tris]
        old_ends = [{x.u, x.v} for x in self.edges] + [{old.u, old.v}]
        return Triangulation._rebuild_from_gluing(
            new_tris, old_ends, fresh_start=self.n_punctures)

    @classmethod
    def _rebuild_from_gluing(
        cls,
        triangle_edges: list[tuple[int, int, int]],
        old_end_labels: list[set[int]],
        *,
        fresh_start: int,
    ) -> "Triangulation":
        """Build a (bordered or closed) id-based triangulation from raw
        triangle-edge data, RE-DERIVING vertex labels from the corner
        orbits of the new gluing: each orbit prefers an unclaimed label
        common to all its corner constraints (intersection), falls back
        to the smallest unclaimed label seen at any of its corners
        (union — the merged-vertex case under gluing), and mints fresh
        labels from `fresh_start` when everything is claimed (the
        split-vertex case under cutting)."""
        n_edges = len(old_end_labels)
        new_inc: list[list[tuple[int, int]]] = [[] for _ in range(n_edges)]
        for ti, te in enumerate(triangle_edges):
            for pos, ei in enumerate(te):
                new_inc[ei].append((ti, pos))
        orbits, _chains = cls._corner_orbits(list(triangle_edges), new_inc)
        orbits = sorted(orbits, key=min)
        cands_int, cands_uni = [], []
        for orb in orbits:
            ci: set[int] | None = None
            cu: set[int] = set()
            for (ti, k) in orb:
                e_in = old_end_labels[triangle_edges[ti][(k + 1) % 3]]
                e_out = old_end_labels[triangle_edges[ti][(k + 2) % 3]]
                local = e_in & e_out
                ci = local if ci is None else (ci & local)
                cu |= local
            cands_int.append(ci or set())
            cands_uni.append(cu)
        label_of: list[int] = [-1] * len(orbits)
        claimed: set[int] = set()
        fresh = fresh_start
        changed = True
        while changed:
            changed = False
            for oi in range(len(orbits)):
                if label_of[oi] >= 0:
                    continue
                free = cands_int[oi] - claimed
                if len(free) == 1:
                    label_of[oi] = free.pop()
                    claimed.add(label_of[oi])
                    changed = True
        for oi in range(len(orbits)):
            if label_of[oi] < 0:
                free = (cands_int[oi] - claimed) or (cands_uni[oi] - claimed)
                if free:
                    label_of[oi] = min(free)
                else:
                    label_of[oi] = fresh
                    fresh += 1
                claimed.add(label_of[oi])
        # compress labels onto 0..V-1 keeping relative order
        order = sorted(set(label_of))
        remap = {l: i for i, l in enumerate(order)}
        label_of = [remap[l] for l in label_of]
        orbit_of: dict[tuple[int, int], int] = {}
        for oi, orb in enumerate(orbits):
            for c in orb:
                orbit_of[c] = oi
        new_edges: list[tuple[int, int]] = []
        for ei in range(n_edges):
            (ti, p) = new_inc[ei][0]
            u = label_of[orbit_of[(ti, (p + 1) % 3)]]
            v = label_of[orbit_of[(ti, (p - 1) % 3)]]
            new_edges.append((min(u, v), max(u, v)))
        return cls.from_edge_data(
            n_punctures=len(orbits),
            edges=new_edges,
            triangle_edges=list(triangle_edges),
            allow_boundary=True,
        )

    def self_glue(self, b1: int, b2: int) -> "Triangulation":
        """GLUE boundary edges `b1` and `b2` of this chart into one
        internal edge (the inverse of `cut`;
        orientation-compatible: the two slots traverse the merged edge
        oppositely).  The merged edge keeps id `min(b1,b2)`; higher edge
        ids shift down by one.  Merged vertices take the smallest of
        the old labels; the result may be bordered or closed (gluing
        the last boundary pair closes the surface)."""
        if b1 == b2:
            raise ValueError("cannot glue a boundary edge to itself")
        for b in (b1, b2):
            if len(self.edge_incidences[b]) != 1:
                raise ValueError(f"edge {b} is not a boundary edge")
        keep, drop = min(b1, b2), max(b1, b2)

        def renum(e):
            if e == drop:
                return keep
            return e - 1 if e > drop else e

        new_tris = [tuple(renum(e) for e in te)
                    for te in self.triangle_edges]
        old_ends: list[set[int]] = []
        for ei, e in enumerate(self.edges):
            if ei == drop:
                continue
            s = {e.u, e.v}
            if ei == keep:
                d = self.edges[drop]
                s = s | {d.u, d.v}
            old_ends.append(s)
        return Triangulation._rebuild_from_gluing(
            new_tris, old_ends, fresh_start=self.n_punctures)

    @staticmethod
    def glue(t1: "Triangulation", b1: int,
             t2: "Triangulation", b2: int) -> "Triangulation":
        """Glue boundary edge `b1` of `t1` to boundary edge `b2` of
        `t2` (disjoint charts), returning the connected result.  `t2`'s
        edge ids are appended after `t1`'s (so its edge j becomes
        `t1.n_edges + j`); further pairs can then be glued with
        `self_glue`.  This is the assembly move of the cut-into-
        polygons / glue-back programme."""
        for t, b in ((t1, b1), (t2, b2)):
            if len(t.edge_incidences[b]) != 1:
                raise ValueError(f"edge {b} is not a boundary edge")
        off_e = t1.n_edges
        off_v = t1.n_punctures
        tris = [tuple(te) for te in t1.triangle_edges] + [
            tuple(e + off_e for e in te) for te in t2.triangle_edges]
        ends = [{e.u, e.v} for e in t1.edges] + [
            {e.u + off_v, e.v + off_v} for e in t2.edges]
        n_tris1 = len(t1.triangle_edges)
        combined = Triangulation.__new__(Triangulation)
        # build a light shim carrying just what self_glue needs
        object.__setattr__(combined, "triangle_edges", tris)
        object.__setattr__(combined, "edges",
                           [Edge(min(s), max(s)) for s in ends])
        inc: list[list[tuple[int, int]]] = [[] for _ in ends]
        for ti, te in enumerate(tris):
            for pos, ei in enumerate(te):
                inc[ei].append((ti, pos))
        object.__setattr__(combined, "edge_incidences", inc)
        object.__setattr__(combined, "n_punctures",
                           t1.n_punctures + t2.n_punctures)
        del n_tris1
        return combined.self_glue(b1, b2 + off_e)

    @classmethod
    def from_bordered_slots(cls, bt) -> "Triangulation":
        """Convert a slot-based `BorderedTriangulation` (which carries
        vertex tuples) into the id-based bordered representation."""
        slot_to_edge, _internal, _boundary = bt.edge_numbering()
        tris = [tuple(slot_to_edge[(ti, pos)] for pos in range(3))
                for ti in range(bt.n_triangles)]
        n_edges = bt.n_edges
        endpoints: list[tuple[int, int] | None] = [None] * n_edges
        for ti, verts in enumerate(bt.triangle_verts):
            for pos in range(3):
                u, v = verts[pos], verts[(pos + 1) % 3]
                endpoints[slot_to_edge[(ti, pos)]] = (min(u, v), max(u, v))
        n_verts = 1 + max(v for ve in bt.triangle_verts for v in ve)
        return cls.from_edge_data(
            n_punctures=n_verts,
            edges=[p for p in endpoints],
            triangle_edges=tris,
            allow_boundary=True,
        )

    # --- the diagonal flip (id-based edge refactor) --------

    def puncture_at_corner(self, ti: int, k: int) -> int:
        """The puncture at corner `k` of triangle `ti` (the vertex between
        the edges at ccw slots k+1 and k+2).  Strict path: `verts[(k+2)%3]`;
        relaxed path: the corner's orbit index (= nominal label, aligned at
        construction)."""
        if self.puncture_orbits is not None:
            for p, orb in enumerate(self.puncture_orbits):
                if (ti, k) in orb:
                    return p
            raise AssertionError(f"corner ({ti},{k}) in no orbit")
        return self.triangles[ti].verts[(k + 2) % 3]

    def flip(self, edge: int) -> "Triangulation":
        """The diagonal flip at internal edge `edge`.

        The two adjacent triangles, rotated to put the flipped edge
        first, read ccw as (e, a, b) and (e, c, d); the quadrilateral's
        ccw boundary is a, b, c, d, and the flip replaces `e` by the
        other diagonal `f`, giving triangles (f, b, c) and (f, d, a).
        The new edge REUSES the flipped edge's index, so the flip is
        index-aligned with quiver mutation at node `edge`:

            flip_e(Delta).sigma()  ==  mu_e(Delta.sigma())

        (the Fomin-Shapiro-Thurston flip/mutation compatibility; pinned
        empirically in `tests/test_triangulation_flip.py` over every
        edge of every fixture, both construction paths).

        Flips routinely CREATE multi-edges and self-loops even from
        strict charts (e.g. any tetrahedron flip), so the result is
        always built through `from_edge_data`.  A flip that would
        create a self-folded triangle (b == c or d == a — the
        once-punctured-monogon configuration) honest-fails.
        """
        inc = self.edge_incidences[edge]
        if len(inc) != 2:
            raise ValueError(f"edge {edge} is not internal (incidences {inc})")
        (t1, p1), (t2, p2) = inc
        te1, te2 = self.triangle_edges[t1], self.triangle_edges[t2]
        a, b = te1[(p1 + 1) % 3], te1[(p1 + 2) % 3]
        c, d = te2[(p2 + 1) % 3], te2[(p2 + 2) % 3]
        if b == c or d == a:
            raise NotImplementedError(
                f"flip at edge {edge} would create a self-folded triangle "
                f"(quadrilateral sides a={a}, b={b}, c={c}, d={d}); "
                "self-folded charts are out of engine scope"
            )
        # the new diagonal joins the corners opposite `edge` in t1, t2
        u = self.puncture_at_corner(t1, p1)
        v = self.puncture_at_corner(t2, p2)
        new_edges = [(e.u, e.v) for e in self.edges]
        new_edges[edge] = (min(u, v), max(u, v))
        new_tris = [tuple(te) for te in self.triangle_edges]
        new_tris[t1] = (edge, b, c)
        new_tris[t2] = (edge, d, a)
        return Triangulation.from_edge_data(
            n_punctures=self.n_punctures,
            edges=new_edges,
            triangle_edges=new_tris,
            allow_boundary=self.is_bordered,
        )

    def canonical_form(self) -> tuple:
        """Canonical encoding of the triangle-edge gluing structure, up
        to orientation-preserving combinatorial isomorphism (edge /
        triangle relabelling; puncture labels ignored).  Two
        triangulations are combinatorially isomorphic iff their
        canonical forms are equal: the encoding is the minimum, over
        every (start triangle, rotation), of the BFS discovery-order
        relabelling of `triangle_edges` — well-defined because the
        surface is connected."""
        tris = self.triangle_edges
        best = None
        for t0 in range(len(tris)):
            for rot in range(3):
                edge_lab: dict[int, int] = {}
                tri_lab = {t0: 0}
                start_rot = {t0: rot}
                from collections import deque
                queue = deque([t0])
                order = []

                def elab(e):
                    if e not in edge_lab:
                        edge_lab[e] = len(edge_lab)
                    return edge_lab[e]

                while queue:
                    ti = queue.popleft()
                    r = start_rot[ti]
                    te = tris[ti]
                    order.append(tuple(elab(te[(r + i) % 3]) for i in range(3)))
                    for i in range(3):
                        e = te[(r + i) % 3]
                        for (tj, pj) in self.edge_incidences[e]:
                            if tj == ti and pj == (r + i) % 3:
                                continue
                            if tj not in tri_lab:
                                tri_lab[tj] = len(tri_lab)
                                start_rot[tj] = pj
                                queue.append(tj)
                enc = tuple(order)
                if best is None or enc < best:
                    best = enc
        return best

    # --- sigma exchange matrix ----

    def sigma(self) -> list[list[int]]:
        """The exchange matrix sigma_Delta as an n_edges x n_edges Z-matrix.

        sigma(e, f) = (# triangles in which f follows e ccw)
                    - (# triangles in which e follows f ccw)

        "f follows e ccw in triangle T" means that in T's ccw edge cycle
        e -> f -> g -> e (or e -> g -> f -> e, etc.), f comes immediately
        after e.
        """
        n = self.n_edges
        s = [[0] * n for _ in range(n)]
        for te in self.triangle_edges:
            for pos in range(3):
                e_i = te[pos]
                e_j = te[(pos + 1) % 3]
                s[e_i][e_j] += 1
                s[e_j][e_i] -= 1
        return s

    # --- standard fixtures -------------------------------------------

    @classmethod
    def from_edge_data(
        cls,
        n_punctures: int,
        edges: list[tuple[int, int]],
        triangle_edges: list[tuple[int, int, int]],
        *,
        allow_boundary: bool = False,
    ) -> "Triangulation":
        """Construct a Triangulation directly from edge-index data.

        Replaces the strict constructor's vertex-tuple checks with
        gluing-derived ones (id-based edge refactor).
        Supports:

          * self-loop edges (v1 == v2), e.g. the bifundamental self-loop
            in the cluster code's `SU2_chain_flavoured(N, 2, 2)` and all
            edges of the once-punctured torus;
          * multi-edges (multiple Edge instances with the same (v1, v2));

        but rejects **self-folded triangles** in the strict FST sense:
        within each triangle the 3 edge indices must be distinct (no
        edge glued to itself within one triangle) -- see the module
        docstring for why that case is genuinely deeper.

        Inputs:
            n_punctures      number of punctures (vertices).
            edges            list of (v1, v2) endpoint pairs.  Self-
                             loops (v1 == v2) and multiple edges with
                             the same (v1, v2) pair are allowed.  The
                             labels are validated against the corner
                             orbits of the actual gluing (a unique
                             orbit <-> label assignment must exist).
            triangle_edges   list of (e0, e1, e2) edge-index triples,
                             one per triangle, in CCW position order.
                             The 3 indices in each triple must be
                             distinct (no self-folded triangles).

        Derived on construction: `slot_forward` (per-slot orientation
        bits — the first incidence of each edge traverses it forward,
        the second backward: the orientable completion of edge-only
        gluing data), `puncture_orbits` (corners grouped by the walk
        around each vertex, aligned to the nominal labels), and
        `genus` via V - E + F.  The resulting object has empty
        `triangles` and `edge_index` (vertex tuples are not derivable
        and endpoint lookup is ambiguous for multi-edges); every
        engine consumes `triangle_edges` / `edge_incidences` /
        `slot_forward` instead.
        """
        # Build Edge instances (allowing self-loops).
        edge_objs = [Edge(min(u, v), max(u, v)) for (u, v) in edges]

        # Validate every triangle edge index is in range AND distinct
        # within each triangle (no strict self-folded triangles).
        n_edges = len(edge_objs)
        for ti, (a, b, c) in enumerate(triangle_edges):
            for ei in (a, b, c):
                if not (0 <= ei < n_edges):
                    raise ValueError(
                        f"triangle {ti} references edge index {ei} "
                        f"out of range [0, {n_edges})"
                    )
            if len({a, b, c}) != 3:
                raise ValueError(
                    f"triangle {ti} has repeated edge index in {(a, b, c)}; "
                    "self-folded triangles (an edge glued to itself) "
                    "are not supported"
                )

        # incidences; a closed surface needs exactly two per edge, a
        # bordered one allows BOUNDARY edges (single incidence = the
        # unglued marked intervals of irregular punctures)
        edge_incidences: list[list[tuple[int, int]]] = [[] for _ in edge_objs]
        for ti, te in enumerate(triangle_edges):
            for pos, ei in enumerate(te):
                edge_incidences[ei].append((ti, pos))
        allowed = (1, 2) if allow_boundary else (2,)
        for ei, inc in enumerate(edge_incidences):
            if len(inc) not in allowed:
                raise ValueError(
                    f"edge {ei} appears in {len(inc)} triangle slots "
                    + ("(expected 1 or 2 for a bordered triangulation)"
                       if allow_boundary else
                       "(expected 2 for a closed triangulation)")
                )
        boundary_ids = [ei for ei, inc in enumerate(edge_incidences)
                        if len(inc) == 1]

        # Orientation bits: first incidence forward, second backward
        # (a boundary edge's single incidence is forward).
        slot_forward_mut: list[list[bool]] = [
            [True, True, True] for _ in triangle_edges
        ]
        for ei, inc in enumerate(edge_incidences):
            if len(inc) == 1:
                (t1, p1), = inc
                slot_forward_mut[t1][p1] = True
                continue
            (t1, p1), (t2, p2) = inc
            slot_forward_mut[t1][p1] = True
            slot_forward_mut[t2][p2] = False
        slot_forward = [tuple(bits) for bits in slot_forward_mut]

        # Vertices = corner orbits (cycles = interior punctures, chains
        # = boundary marked points); align them with the nominal labels.
        orbits, chain_flags = cls._corner_orbits(
            list(triangle_edges), edge_incidences)
        if len(orbits) != n_punctures:
            raise ValueError(
                f"gluing has {len(orbits)} vertices (corner orbits/chains) "
                f"but n_punctures = {n_punctures}"
            )
        # Candidate labels per orbit: every corner of the orbit lies at
        # an endpoint of both its incoming and outgoing edges.
        cands: list[set[int]] = []
        for orb in orbits:
            cand: set[int] | None = None
            for (ti, k) in orb:
                e_in = edge_objs[triangle_edges[ti][(k + 1) % 3]]
                e_out = edge_objs[triangle_edges[ti][(k + 2) % 3]]
                local = {e_in.u, e_in.v} & {e_out.u, e_out.v}
                cand = local if cand is None else (cand & local)
            if not cand:
                raise ValueError(
                    "edge endpoint labels are inconsistent with the "
                    "gluing: a corner orbit has no common endpoint "
                    "label"
                )
            cands.append(set(cand))
        # Unique-assignment elimination (orbit <-> label bijection).
        assigned: dict[int, int] = {}   # orbit index -> label
        claimed: set[int] = set()
        while len(assigned) < len(orbits):
            progress = False
            for oi, cand in enumerate(cands):
                if oi in assigned:
                    continue
                free = cand - claimed
                if len(free) == 0:
                    raise ValueError(
                        "edge endpoint labels are inconsistent with the "
                        "gluing: no label left for a corner orbit"
                    )
                if len(free) == 1:
                    lbl = next(iter(free))
                    assigned[oi] = lbl
                    claimed.add(lbl)
                    progress = True
            if not progress:
                raise ValueError(
                    "edge endpoint labels do not determine a unique "
                    "orbit <-> puncture assignment (ambiguous labels); "
                    "use the strict Triangle-based constructor or "
                    "distinct labels"
                )
        puncture_orbits: list[list[tuple[int, int]] | None] = [None] * n_punctures
        marked = []
        for oi, lbl in assigned.items():
            if not (0 <= lbl < n_punctures):
                raise ValueError(
                    f"orbit assigned out-of-range puncture label {lbl}"
                )
            puncture_orbits[lbl] = orbits[oi]
            if chain_flags[oi]:
                marked.append(lbl)

        # Holes = boundary circles: from boundary edge e (incidence
        # (t, p)), the corner whose INCOMING slot is p walks around the
        # marked point; the chain's terminal corner's OUTGOING edge is
        # the next boundary edge around the hole.
        holes: list[tuple[int, ...]] = []
        if boundary_ids:
            def next_boundary(e):
                (ti, p), = edge_incidences[e]
                cur = (ti, (p - 1) % 3)      # corner with incoming slot p
                while True:
                    pos_out = (cur[0], (cur[1] + 2) % 3)
                    e_out = triangle_edges[cur[0]][pos_out[1]]
                    inc = edge_incidences[e_out]
                    if len(inc) == 1:
                        return e_out
                    (t1, p1), (t2, p2) = inc
                    tj, pj = (t2, p2) if (t1, p1) == pos_out else (t1, p1)
                    cur = (tj, (pj - 1) % 3)
            left = set(boundary_ids)
            while left:
                start = min(left)
                cyc, cur = [], start
                while True:
                    cyc.append(cur)
                    left.discard(cur)
                    cur = next_boundary(cur)
                    if cur == start:
                        break
                holes.append(tuple(cyc))

        # Euler relation -> genus (V from the gluing, not the labels).
        # Closed: chi = 2 - 2g.  Bordered: chi = 2 - 2g - #holes.
        chi = n_punctures - n_edges + len(triangle_edges)
        two_g = 2 - chi - len(holes)
        if two_g < 0 or two_g % 2 != 0:
            raise ValueError(
                f"Euler characteristic V-E+F = {chi} with {len(holes)} "
                "holes: not of the form 2 - 2g - b for an orientable "
                "surface"
            )

        # Construct the dataclass bypassing __post_init__'s strict checks.
        obj = cls.__new__(cls)
        object.__setattr__(obj, "n_punctures", n_punctures)
        object.__setattr__(obj, "triangles", [])  # not used in this path
        object.__setattr__(obj, "edges", edge_objs)
        object.__setattr__(obj, "edge_index", {})  # not unique for multi-edges
        object.__setattr__(obj, "triangle_edges", list(triangle_edges))
        object.__setattr__(obj, "edge_incidences", edge_incidences)
        object.__setattr__(obj, "slot_forward", slot_forward)
        object.__setattr__(obj, "puncture_orbits", puncture_orbits)
        object.__setattr__(obj, "marked_points", tuple(sorted(marked)))
        object.__setattr__(obj, "holes", tuple(holes))
        object.__setattr__(obj, "genus", two_g // 2)
        return obj

    @classmethod
    def cluster_sausage_S2_5(cls) -> "Triangulation":
        """The cluster code's `SU2_chain_flavoured(2, left_nf=2, right_nf=2)`
        triangulation of S^2_{0,5}, expressed as edge-index data.

        Has self-loops (e_4 is at puncture 2; per the cluster docstring
        this represents the bifundamental matter) and multi-edges
        (e_0, e_1 both connect punctures 0--2; e_2, e_3 both connect
        2--3).  Every triangle has 3 *distinct* edge indices -- no
        self-folded triangles in the strict FST sense, matching the
        cluster A1 constructor's deliberate avoidance of those.

        Sigma() on this triangulation matches `class_S_A1_bps(5, [])`
        BPS quiver exchange matrix entry-by-entry.  Use the
        from_edge_data path: the standard Triangle-based constructor
        cannot represent self-loops or multi-edges.

        Endpoint labels are consistent with the triangle gluing: the
        corner-orbit walk on (triangle_edges, edge_incidences) yields
        exactly these five punctures, and every puncture_flavour_charge
        lies in ker(sigma).
        """
        return cls.from_edge_data(
            n_punctures=5,
            edges=[
                (0, 2),  # e_0
                (0, 2),  # e_1 (multi-edge with e_0)
                (2, 3),  # e_2
                (2, 3),  # e_3 (multi-edge with e_2)
                (2, 2),  # e_4 (self-loop at puncture 2; bifundamental)
                (1, 2),  # e_5
                (0, 1),  # e_6
                (2, 4),  # e_7
                (3, 4),  # e_8
            ],
            triangle_edges=[
                (0, 1, 4),  # T_0: 3 distinct edges, vertices {0, 2}
                (2, 3, 4),  # T_1: 3 distinct edges, vertices {2, 3}
                (0, 5, 6),  # T_2
                (6, 5, 1),  # T_3
                (2, 7, 8),  # T_4
                (8, 7, 3),  # T_5
            ],
        )

    @classmethod
    def csaszar_torus_T2_7(cls) -> "Triangulation":
        """The Csaszar (K_7) ideal triangulation of the 7-punctured torus.

        The complete graph K_7 embeds on the torus with all 14 faces
        triangles: vertices Z_7, faces (i, i+1, i+3) and (i, i+3, i+2)
        (orientations chosen so each edge is traversed once each way).
        Every vertex pair carries exactly one edge, so the strict
        Triangle-based constructor applies -- the smallest non-sphere
        chart with neither self-loops nor multi-edges.  genus == 1.
        """
        tris = []
        for i in range(7):
            tris.append(Triangle(((i) % 7, (i + 1) % 7, (i + 3) % 7)))
            tris.append(Triangle(((i) % 7, (i + 3) % 7, (i + 2) % 7)))
        return cls(n_punctures=7, triangles=tris)

    @classmethod
    def bipyramid_S2_5(cls) -> "Triangulation":
        """The triangular-bipyramid triangulation of S^2_{0,5}.

        Punctures 0,1,2 on the equator, 3 and 4 at the poles; six
        triangles, nine edges, all with distinct endpoints (so the
        strict Triangle-based constructor applies, unlike the
        self-loop `cluster_sausage_S2_5` chart).  Gauge rank 4
        (= SU(2) x SU(2) sausage), flavour rank 5.
        """
        return cls(
            n_punctures=5,
            triangles=[
                Triangle(verts=(0, 1, 3)),
                Triangle(verts=(1, 2, 3)),
                Triangle(verts=(2, 0, 3)),
                Triangle(verts=(1, 0, 4)),
                Triangle(verts=(2, 1, 4)),
                Triangle(verts=(0, 2, 4)),
            ],
        )

    @classmethod
    def octahedron_S2_6(cls) -> "Triangulation":
        """The octahedral triangulation of S^2_{0,6} (SU(2)^3 sausage).

        Equator square 0-1-2-3, apex 4 (top) and 5 (bottom); eight
        triangles, twelve edges, all with distinct endpoints and no
        multi-edges (the equator diagonals 0-2, 1-3 are not edges), so the
        strict Triangle-based constructor applies.  Gauge rank 3
        (= SU(2)^3), flavour rank 6 -- the next multi-node case after the
        bipyramid (SU(2)^2)."""
        return cls(
            n_punctures=6,
            triangles=[
                Triangle(verts=(0, 1, 4)),
                Triangle(verts=(1, 2, 4)),
                Triangle(verts=(2, 3, 4)),
                Triangle(verts=(3, 0, 4)),
                Triangle(verts=(1, 0, 5)),
                Triangle(verts=(2, 1, 5)),
                Triangle(verts=(3, 2, 5)),
                Triangle(verts=(0, 3, 5)),
            ],
        )

    @classmethod
    def tetrahedron_S2_4(cls) -> "Triangulation":
        """The tetrahedral triangulation of S^2_{0,4}.

        Punctures P_1..P_4 are 0-indexed as 0..3.
        Triangle order T_4, T_3, T_2, T_1.
        """
        return cls(
            n_punctures=4,
            triangles=[
                Triangle(verts=(0, 1, 2)),  # T_4: P_1 -> P_2 -> P_3
                Triangle(verts=(1, 0, 3)),  # T_3: P_2 -> P_1 -> P_4
                Triangle(verts=(0, 2, 3)),  # T_2: P_1 -> P_3 -> P_4
                Triangle(verts=(2, 1, 3)),  # T_1: P_3 -> P_2 -> P_4
            ],
        )
