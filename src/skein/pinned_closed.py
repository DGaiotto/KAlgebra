"""Piecewise closed-surface quantum traces from pinned polygon pieces
(extending the pinned tier to closed surfaces).

A closed triangulated surface cut along a subset of edges (the SEAMS)
falls apart into bordered polygon pieces; `PiecewiseClosedTrace`
assembles the quantum trace of a closed simple curve on the closed
surface from the pieces' PINNED arcs (`pinned_polygon.PinnedPolygon`):

    Tr^lq(gamma) = sum_{eps : seam crossings -> {+/-}}
                     project( (x)_pieces  kappa_piece(run arcs) )

— the SELF-GLUING state-sum merge.  The curve, cut at its seam
crossings, becomes a cyclic chain of RUNS (maximal segments inside one
piece); each run evaluates to the piece's pinned arc with the seam
states at its two ends, with all PIECE-INTERNAL crossings already
summed inside the pinned arc (that is the payoff: only seam crossings
carry state variables — 2^{#seam crossings} terms instead of
2^{#crossings} — and the per-piece arcs are computed once and cached
across curves).  Terms are projected to `Y_Delta` by the standard
`project_triangle_monomial` (lift-parity + the ccw->global reorder),
so the result is certified EXACTLY against the closed single-arc
state-sum engine `quantum_trace` (the suite in the source repository: every
seeded single-arc curve of the tetrahedron and the bipyramid, on
several cuts).

The RG reading: every seam is an
adjoined dynamical direction — gluing back the pieces is the inverse
of the node-drop RG that cutting performs, now at the level of the
closed-surface trace: the closed Tr^lq (the geometric F / RG image)
is assembled from IR polygon data.

Scope (honest): closed Triangle-based triangulations; pieces of ANY
size whose induced dual subgraph is a connected TREE (a triangulated
disk — the stated engine and its unique arc routing are tree-dual
general since this landing; pieces with dual cycles, e.g. a full
vertex star, must be cut open); curves in the
single-crossing-per-edge, single-arc-per-triangle normal scope of
`realize_curve`; every curve must cross at least one seam (a curve
contained in a piece would need the piece's own closed trace).
"""

from __future__ import annotations

from itertools import product as iproduct
from typing import Sequence

from curve_realization import realize_curve
from half_laurent import HalfLaurent, ONE as HL_ONE
from pinned_polygon import PinnedPolygon
from triangulation import Triangulation
from y_delta import YDelta, project_triangle_monomial


class PiecewiseClosedTrace:
    """Assemble closed-curve quantum traces from pinned polygon pieces."""

    def __init__(self, t: Triangulation, pieces: Sequence[Sequence[int]]):
        if not getattr(t, "triangles", None):
            raise NotImplementedError(
                "PiecewiseClosedTrace needs a Triangle-based closed "
                "Triangulation")
        self.t = t
        self.pieces = [tuple(p) for p in pieces]
        seen = [ti for p in self.pieces for ti in p]
        if sorted(seen) != list(range(t.n_triangles)):
            raise ValueError("pieces must partition the triangles")
        self._pp_by_n: dict = {}            # BorderedTriangulation -> piece
        self.piece_of_tri: dict[int, int] = {}
        # per piece: (pinned polygon, {target tri: (piece tri index, rot)},
        #             {target edge: piece side}, internal edge ids)
        self._piece_data = []
        for pi, p in enumerate(self.pieces):
            for ti in p:
                self.piece_of_tri[ti] = pi
            self._piece_data.append(self._build_piece(p))
        self.seam_edges = set(range(t.n_edges))
        for (_pp, _tris, _e2s, internal) in self._piece_data:
            self.seam_edges -= internal
        self._arc_cache: dict = {}

    # -- piece construction ------------------------------------------------

    def _pp_by_bt(self, bt) -> PinnedPolygon:
        if bt not in self._pp_by_n:
            self._pp_by_n[bt] = PinnedPolygon(bt)
        return self._pp_by_n[bt]

    def _build_piece(self, p):
        """(pinned polygon, tri correspondence, edge->side map, internal
        edges) for a piece of any size whose induced dual subgraph is a
        connected TREE (a triangulated disk).

        The piece's `BorderedTriangulation` is built directly from the
        target data — piece triangle i = the i-th triangle of `p` with
        the TARGET's ccw role order — so the correspondence carries no
        rotation (rot = 0 throughout; the ordered-monomial reorder cost
        `_rot_shift` stays for generality but vanishes here).
        """
        from bordered_triangulation import BorderedTriangulation

        te = self.t.triangle_edges
        k = len(p)
        gluings, internal = [], set()
        for i in range(k):
            for j in range(i + 1, k):
                shared = set(te[p[i]]) & set(te[p[j]])
                if len(shared) > 1:
                    raise NotImplementedError(
                        f"triangles {p[i]},{p[j]} share {len(shared)} "
                        f"edges (dual multi-edge; not a disk piece)")
                if shared:
                    e = shared.pop()
                    gluings.append(((i, te[p[i]].index(e)),
                                    (j, te[p[j]].index(e))))
                    internal.add(e)
        if len(gluings) != k - 1:
            raise NotImplementedError(
                f"piece {p} has {len(gluings)} internal gluings for "
                f"{k} triangles; its dual is not a tree (disk pieces "
                f"only — cut a cycle open instead)")
        glued = {s for pair in gluings for s in pair}
        boundary = tuple((i, r) for i in range(k) for r in range(3)
                         if (i, r) not in glued)
        verts = tuple(self.t.triangles[ti].verts for ti in p)
        bt = BorderedTriangulation(verts, tuple(gluings), boundary)
        pp = self._pp_by_bt(bt)
        tris = {p[i]: (i, 0) for i in range(k)}
        e2s = {te[p[i]][r]: s for s, (i, r) in enumerate(boundary)}
        return (pp, tris, e2s, internal)

    # -- the curve as a cyclic chain of runs ---------------------------------

    def _cyclic_sequence(self, coords):
        """[(triangle, e_in, e_out)] around the curve (single-crossing
        normal scope)."""
        if any(c not in (0, 1) for c in coords):
            raise NotImplementedError(
                "single crossing per edge only (0/1 normal coordinates)")
        arcs = realize_curve(self.t, coords)
        users: dict[int, list] = {}
        for ti, tlist in enumerate(arcs):
            if not tlist:
                continue
            if len(tlist) != 1:
                raise NotImplementedError("single arc per triangle only")
            a = tlist[0]
            for e in (a.edge_at_eps, a.edge_at_eps_prime):
                users.setdefault(e, []).append(ti)
        n_arcs = sum(1 for tl in arcs if tl)
        start_e = min(users)
        tri = users[start_e][0]
        seq, e_in = [], start_e
        for _ in range(n_arcs):
            a = arcs[tri][0]
            e_out = (a.edge_at_eps if a.edge_at_eps != e_in
                     else a.edge_at_eps_prime)
            seq.append((tri, e_in, e_out))
            nxt = [x for x in users[e_out] if x != tri]
            tri, e_in = nxt[0], e_out
        assert e_in == start_e and tri == users[start_e][0], \
            "curve chain did not close"
        return seq

    def _runs(self, seq):
        """Split the cyclic chain at seam crossings: [(piece index,
        s_in, s_out, e_in, e_out)] with piece-side endpoints."""
        k = next((i for i, (_t, e_in, _e) in enumerate(seq)
                  if e_in in self.seam_edges), None)
        if k is None:
            raise NotImplementedError(
                "curve crosses no seam (contained in one piece)")
        seq = seq[k:] + seq[:k]
        runs = []
        cur = []
        for entry in seq:
            if entry[1] in self.seam_edges and cur:
                runs.append(cur)
                cur = []
            cur.append(entry)
        runs.append(cur)
        out = []
        for cur in runs:
            pi = self.piece_of_tri[cur[0][0]]
            assert all(self.piece_of_tri[ti] == pi for ti, _i, _o in cur), \
                "run leaves its piece without a seam crossing"
            (_pp, _tris, e2s, _int) = self._piece_data[pi]
            e_in, e_out = cur[0][1], cur[-1][2]
            out.append((pi, e2s[e_in], e2s[e_out], e_in, e_out))
        return out

    # -- assembly ----------------------------------------------------------------

    def _piece_arc_keys(self, pi, s_in, s_out, eps_in, eps_out):
        """kappa terms of the piece arc, as (per-piece-triangle key,
        coefficient) pairs (cached)."""
        ck = (pi, s_in, s_out, eps_in, eps_out)
        if ck not in self._arc_cache:
            (pp, _tris, _e2s, _int) = self._piece_data[pi]
            x = pp.arc_F(s_in, s_out, eps_in, eps_out)
            self._arc_cache[ck] = [(pp.key_of_charge(u), c)
                                   for u, c in x.items()]
        return self._arc_cache[ck]

    @staticmethod
    def _rot_shift(triple, r):
        """lq-exponent converting a piece-role-ORDERED monomial
        y_a^{xa} y_b^{xb} y_c^{xc} to the target-slot-ordered form
        after a ccw rotation by r (the exponent triples encode ORDERED
        monomials, so a cyclic rotation is a reorder with the
        universal sigma_T commutation cost)."""
        xa, xb, xc = triple
        if r == 1:                          # (a,b,c) -> (c,a,b)
            return xb * xc - xa * xc
        if r == 2:                          # (a,b,c) -> (b,c,a)
            return xa * xb - xa * xc
        return 0

    def trace(self, coords) -> YDelta:
        """Tr^lq of the closed simple curve with the given normal
        coordinates, assembled from the pieces."""
        t = self.t
        seq = self._cyclic_sequence(coords)
        runs = self._runs(seq)
        m = len(runs)
        # seam state variable per run entry; run i exits into run i+1's
        # entry, so eps_out of run i = eps_in of run (i+1) mod m
        result = YDelta.zero(t)
        id3 = (0, 0, 0)
        for assignment in iproduct((1, -1), repeat=m):
            partials = [({}, HL_ONE)]           # (per-target-tri exps, coef)
            dead = False
            for i, (pi, s_in, s_out, _ei, _eo) in enumerate(runs):
                terms = self._piece_arc_keys(
                    pi, s_in, s_out, assignment[i], assignment[(i + 1) % m])
                if not terms:
                    dead = True
                    break
                (_pp, tris, _e2s, _int) = self._piece_data[pi]
                new = []
                for exps, coef in partials:
                    for key, c in terms:
                        cur = dict(exps)
                        cc = coef * c
                        ok = True
                        for ti, (ptri, rot) in tris.items():
                            triple = key[ptri]
                            if triple == id3:
                                continue
                            if ti in cur:       # single-visit scope
                                ok = False
                                break
                            rt = [0, 0, 0]
                            for pos in range(3):
                                rt[(pos + rot) % 3] = triple[pos]
                            cur[ti] = tuple(rt)
                            cc = cc.shift(self._rot_shift(triple, rot))
                        if ok:
                            new.append((cur, cc))
                partials = new
                if not partials:
                    dead = True
                    break
            if dead:
                continue
            for exps, coef in partials:
                triangle_exps = [exps.get(ti, id3)
                                 for ti in range(t.n_triangles)]
                proj = project_triangle_monomial(t, triangle_exps, coef)
                if proj is None:
                    continue
                charge, ocoef = proj
                result = result + YDelta.monomial(t, charge, ocoef)
        return result
