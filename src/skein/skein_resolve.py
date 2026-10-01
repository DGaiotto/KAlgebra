"""Intrinsic skein product: stack one multicurve above another and
resolve all crossings by the Kauffman bracket relations.

This implements the *mathematical construction* of the SL_2 skein
algebra Sk(S^2_{0,n}): elements are Z[A^{+/-1}]-combinations of simple
multicurves (in normal coordinates w.r.t. a fixed ideal triangulation,
see `multicurve.py`); the product places the first factor ABOVE the
second in `surface x [0,1]` and resolves:

    crossing  =  A * (smoothing I)  +  A^{-1} * (smoothing II)
    trivial loop  =  delta  =  -A^2 - A^{-2}

The triangulation plays no structural role -- it is only a coordinate
system: products are computed by gluing exact-rational chord pictures
inside each triangle, smoothing, and re-tightening to normal position
(`MatchingDiagram.normalize`).  The output decomposes over multicurve
diagrams; peripheral components and Chebyshev (bracelet) re-expression
are handled downstream by the K-algebra layer.

Crossing conventions
--------------------
At a transverse crossing the four half-edges alternate over / under in
cyclic order around the crossing.  Smoothings, stated orientation-free:

    "ccw":  pair each OVER half-edge with the under half-edge
            immediately NEXT in counterclockwise order;
    "cw":   the mirror pairing.

With the over-strand SW->NE and under-strand SE->NW (standard Kauffman
picture), "ccw" is the A-smoothing.  Which of the two receives weight
`A` is the `a_smoothing` parameter ("ccw" by default); the choice is
pinned against the quantum-trace oracle in the tests.

Exact geometry
--------------
Inside a triangle, boundary points are placed on the convex parabola
`(t, t^2)` in their ccw boundary order, and arcs are straight chords.
All intersection tests and orderings use exact `Fraction` arithmetic.
If three chords meet at a point (possible for symmetric inputs), the
parameter spacing is perturbed deterministically and the layout
retried.

Self-contained: depends only on `triangulation` and
`multicurve`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from multicurve import (
    MatchingDiagram,
    is_admissible,
    normal_arcs,
)
from triangulation import Triangulation


Coords = tuple[int, ...]
End = tuple[int, int]            # (edge_id, joint point index)
Pt = tuple[Fraction, Fraction]


# ---------------------------------------------------------------------
# Exact 2d helpers
# ---------------------------------------------------------------------


def _sub(p: Pt, q: Pt) -> Pt:
    return (p[0] - q[0], p[1] - q[1])


def _cross(u: Pt, v: Pt) -> Fraction:
    return u[0] * v[1] - u[1] * v[0]


def _seg_intersection(p1: Pt, q1: Pt, p2: Pt, q2: Pt) -> tuple[Fraction, Fraction, Pt] | None:
    """Proper intersection of open segments p1q1, p2q2.

    Returns (s, t, X) with X = p1 + s*(q1-p1) = p2 + t*(q2-p2),
    0 < s, t < 1; or None if they do not properly cross.
    """
    d1 = _sub(q1, p1)
    d2 = _sub(q2, p2)
    denom = _cross(d1, d2)
    if denom == 0:
        return None
    w = _sub(p2, p1)
    s = _cross(w, d2) / denom
    t = _cross(w, d1) / denom
    if not (0 < s < 1 and 0 < t < 1):
        return None
    x = (p1[0] + s * d1[0], p1[1] + s * d1[1])
    return (s, t, x)


def _angle_key(v: Pt) -> tuple[int, Fraction, Fraction]:
    """Sort key for exact ccw angular order of nonzero vectors.

    Half-plane index first (upper half-plane [0, pi) before lower),
    then by slope comparison via cross products: within a half-plane,
    u before v iff cross(u, v) > 0 fails... we instead sort by
    (half, -x/y ...) -- simplest exact scheme: half-plane index, then
    compare two vectors u, v in the same half-plane by cross(u, v) > 0
    meaning u comes first.  Implemented via a key trick: use the
    projective slope y/x ordered appropriately per half-plane.
    """
    x, y = v
    if y > 0 or (y == 0 and x > 0):
        half = 0
    else:
        half = 1
    # Within a half-plane the ccw order is by increasing angle; for
    # half 0 (angles in [0, pi)) angle increases as slope y/x goes
    # 0+ -> +inf (x>0), then jumps to -inf -> 0- (x<0).  Encode by
    # (sign bucket, slope) pairs that sort correctly.
    if half == 0:
        if x > 0:
            return (0, Fraction(0), y / x) if y == 0 else (0, Fraction(1), y / x)
        elif x == 0:
            return (0, Fraction(2), Fraction(0))
        else:
            return (0, Fraction(3), y / x)
    else:
        if x < 0:
            return (1, Fraction(0), y / x) if y == 0 else (1, Fraction(1), y / x)
        elif x == 0:
            return (1, Fraction(2), Fraction(0))
        else:
            return (1, Fraction(3), y / x)


# ---------------------------------------------------------------------
# Per-triangle chord layout
# ---------------------------------------------------------------------


@dataclass
class _Chord:
    """A straight chord inside one triangle.

    `which` is 0 for the first (upper) multicurve, 1 for the second.
    `ends` are the two boundary ends as (edge, joint point) pairs.
    Geometric endpoints are filled by the layout.
    """
    which: int
    ends: tuple[End, End]
    p: Pt = (Fraction(0), Fraction(0))
    q: Pt = (Fraction(0), Fraction(0))


class _LayoutDegenerate(Exception):
    pass


def _triangle_boundary_sequence(
    t: Triangulation, ti: int, n_points: Sequence[int],
) -> list[End]:
    """All joint boundary points of triangle `ti` in ccw boundary order.

    The edge at ccw position `pos` is traversed in the triangle's ccw
    direction; its points (indexed along the edge's canonical
    direction) are listed in traversal order — forward when the slot's
    orientation bit is set, reversed otherwise.  Consumes only
    `slot_forward`, so self-loops and multi-edges work (id-based edge
    refactor, 2026-07-07).
    """
    seq: list[End] = []
    for pos in range(3):
        e_id = t.triangle_edges[ti][pos]
        n = n_points[e_id]
        if t.slot_forward[ti][pos]:
            idx = range(n)
        else:
            idx = range(n - 1, -1, -1)
        for pidx in idx:
            seq.append((e_id, pidx))
    return seq


def _layout_chords(
    boundary_seq: list[End],
    chords: list[_Chord],
    seed: int,
) -> list[tuple[int, int, Fraction, Fraction, Pt]]:
    """Place boundary points on the parabola and intersect all chords.

    Returns the list of proper crossings as tuples
    `(chord_i, chord_j, s_i, s_j, X)` (i < j), after checking generic
    position (no coincident crossing points on a chord, no crossing at
    equal parameters).  Raises _LayoutDegenerate to request a retry
    with a different seed.
    """
    n = len(boundary_seq)
    pos_of: dict[End, Pt] = {}
    for i, end in enumerate(boundary_seq):
        # deterministic perturbation: t_i = i + seed * i^2 / (large)
        ti_param = Fraction(i) + Fraction(seed * (i * i + 1), 9973 * (n + 7))
        pos_of[end] = (ti_param, ti_param * ti_param)
    for c in chords:
        c.p = pos_of[c.ends[0]]
        c.q = pos_of[c.ends[1]]
    crossings: list[tuple[int, int, Fraction, Fraction, Pt]] = []
    seen_points: dict[int, list[Pt]] = {}
    for i in range(len(chords)):
        for j in range(i + 1, len(chords)):
            ci, cj = chords[i], chords[j]
            hit = _seg_intersection(ci.p, ci.q, cj.p, cj.q)
            if hit is None:
                continue
            if ci.which == cj.which:
                raise AssertionError(
                    "two chords of the same multicurve cross; normal "
                    "arcs must be disjoint (layout or input bug)"
                )
            s, tt, x = hit
            for k in (i, j):
                for prev in seen_points.get(k, []):
                    if prev == x:
                        raise _LayoutDegenerate()
                seen_points.setdefault(k, []).append(x)
            crossings.append((i, j, s, tt, x))
    return crossings


# ---------------------------------------------------------------------
# Resolution engine
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedTerm:
    """One smoothing outcome of the stacked diagram, tightened.

    coords     normal coordinates of the resulting multicurve
    n_trivial  number of contractible loops produced (each worth delta)
    a_exponent net Kauffman exponent: weight A^{a_exponent}
    """
    coords: Coords
    n_trivial: int
    a_exponent: int


def resolve_product(
    t: Triangulation,
    coords_above: Sequence[int],
    coords_below: Sequence[int],
    *,
    a_smoothing: str = "ccw",
) -> list[ResolvedTerm]:
    """All Kauffman smoothings of (above stacked over below), tightened.

    The two multicurves are put in joint position: on every edge, the
    `above` points come first along the edge's canonical orientation,
    then the `below` points.  Crossings happen only inside triangles.

    Returns one `ResolvedTerm` per smoothing assignment (2^c terms for
    c crossings; no combining).  The caller multiplies each term by
    `A^{a_exponent} * delta^{n_trivial}` and decomposes `coords`.
    """
    if a_smoothing not in ("ccw", "cw"):
        raise ValueError("a_smoothing must be 'ccw' or 'cw'")
    a = tuple(coords_above)
    b = tuple(coords_below)
    if not is_admissible(t, a) or not is_admissible(t, b):
        raise ValueError("both factors must be admissible normal coords")

    n_edges = t.n_edges
    joint_points = [a[e] + b[e] for e in range(n_edges)]

    # Joint point indexing on edge e: 0..a_e-1 are the ABOVE curve's
    # points (in its own canonical order), a_e..a_e+b_e-1 are BELOW's.
    arcs_above = normal_arcs(t, a)
    arcs_below = normal_arcs(t, b)

    def lift_end(end: End, which: int) -> End:
        e, p = end
        return (e, p if which == 0 else a[e] + p)

    # Build per-triangle chords over joint points.
    tri_chords: list[list[_Chord]] = []
    for ti in range(t.n_triangles):
        chords: list[_Chord] = []
        for arc in arcs_above[ti]:
            chords.append(_Chord(0, (lift_end(arc.end_a, 0), lift_end(arc.end_b, 0))))
        for arc in arcs_below[ti]:
            chords.append(_Chord(1, (lift_end(arc.end_a, 1), lift_end(arc.end_b, 1))))
        tri_chords.append(chords)

    # Lay out and collect crossings per triangle.
    tri_crossings: list[list[tuple[int, int, Fraction, Fraction, Pt]]] = []
    for ti in range(t.n_triangles):
        seq = _triangle_boundary_sequence(t, ti, joint_points)
        for seed in range(1, 50):
            try:
                crossings = _layout_chords(seq, tri_chords[ti], seed)
                break
            except _LayoutDegenerate:
                continue
        else:
            raise RuntimeError(f"no generic layout found for triangle {ti}")
        tri_crossings.append(crossings)

    # Global crossing list: (triangle, local crossing index).
    all_crossings = [
        (ti, k) for ti in range(t.n_triangles) for k in range(len(tri_crossings[ti]))
    ]
    n_cross = len(all_crossings)

    # Pre-compute, per triangle, the segment structure of each chord:
    # crossings along the chord sorted by parameter, with the crossing's
    # global index and the local cyclic geometry at the crossing.
    #
    # Per crossing we store the two pairings as frozensets of half-edge
    # ids; half-edges are (chord_index, 'in'|'out') at that crossing.
    @dataclass
    class _CrossInfo:
        ti: int
        chord_i: int          # ABOVE chord (which == 0)
        chord_j: int          # BELOW chord
        s_i: Fraction
        s_j: Fraction
        pairing_ccw: tuple[tuple[tuple[int, str], tuple[int, str]], ...]
        pairing_cw: tuple[tuple[tuple[int, str], tuple[int, str]], ...]

    cross_info: list[_CrossInfo] = []
    for gidx, (ti, k) in enumerate(all_crossings):
        i, j, s, tt, x = tri_crossings[ti][k]
        ci, cj = tri_chords[ti][i], tri_chords[ti][j]
        if ci.which == 0:
            over_idx, under_idx = i, j
            s_over, s_under = s, tt
            over, under = ci, cj
        else:
            over_idx, under_idx = j, i
            s_over, s_under = tt, s
            over, under = cj, ci
        # Half-edge direction vectors at X: 'in' points back toward the
        # chord start p, 'out' toward q.
        dirs = {
            (over_idx, "in"): _sub(over.p, x),
            (over_idx, "out"): _sub(over.q, x),
            (under_idx, "in"): _sub(under.p, x),
            (under_idx, "out"): _sub(under.q, x),
        }
        order = sorted(dirs, key=lambda h: _angle_key(dirs[h]))
        # ccw rule: each OVER half-edge pairs with the under half-edge
        # immediately next in ccw order.
        def _pair(rule_ccw: bool):
            pairs = []
            m = len(order)
            for pos_, h in enumerate(order):
                if h[0] != over_idx:
                    continue
                nxt = order[(pos_ + 1) % m] if rule_ccw else order[(pos_ - 1) % m]
                if nxt[0] != under_idx:
                    raise AssertionError(
                        "over/under half-edges do not alternate at crossing"
                    )
                pairs.append((h, nxt))
            return tuple(pairs)

        cross_info.append(_CrossInfo(
            ti=ti, chord_i=over_idx, chord_j=under_idx,
            s_i=s_over, s_j=s_under,
            pairing_ccw=_pair(True), pairing_cw=_pair(False),
        ))

    # Per triangle, per chord: ordered crossing list along the chord.
    chord_crossings: dict[tuple[int, int], list[tuple[Fraction, int]]] = {}
    for gidx, info in enumerate(cross_info):
        chord_crossings.setdefault((info.ti, info.chord_i), []).append((info.s_i, gidx))
        chord_crossings.setdefault((info.ti, info.chord_j), []).append((info.s_j, gidx))
    for key in chord_crossings:
        chord_crossings[key].sort()

    # Node graph per triangle: nodes are boundary ends ('B', end) and
    # crossing half-points ('X', global_idx).  Each chord contributes a
    # path: end0 -- X_{(1)} -- X_{(2)} -- ... -- end1, where the X's are
    # its crossings in order.  At each crossing, the smoothing chooses
    # how the four incident path-segments reconnect.
    #
    # We encode each chord as the list of its 'stations':
    #   [('B', end0), ('X', g1), ('X', g2), ..., ('B', end1)]
    # Segment k of the chord joins stations k and k+1.  At a crossing
    # station ('X', g) of chord c, the two adjacent segments are the
    # 'in' side (toward end0) and 'out' side (toward end1).
    tri_segments: list[dict] = []
    for ti in range(t.n_triangles):
        stations: dict[int, list] = {}
        for c_idx in range(len(tri_chords[ti])):
            xs = chord_crossings.get((ti, c_idx), [])
            st = [("B", tri_chords[ti][c_idx].ends[0])]
            for _s, g in xs:
                st.append(("X", g))
            st.append(("B", tri_chords[ti][c_idx].ends[1]))
            stations[c_idx] = st
        tri_segments.append(stations)

    results: list[ResolvedTerm] = []

    for mask in range(1 << n_cross):
        # smoothing[g] = 'ccw' or 'cw'
        a_exp = 0
        pair_at: dict[int, dict[tuple[int, str], tuple[int, str]]] = {}
        for g, info in enumerate(cross_info):
            use_ccw = bool((mask >> g) & 1)
            chosen = info.pairing_ccw if use_ccw else info.pairing_cw
            rule_is_a = (a_smoothing == "ccw") == use_ccw
            a_exp += 1 if rule_is_a else -1
            m: dict[tuple[int, str], tuple[int, str]] = {}
            for (h_over, h_under) in chosen:
                m[h_over] = h_under
                m[h_under] = h_over
            pair_at[g] = m

        # Walk the smoothed strands in each triangle.
        n_loops = 0
        arcs_per_triangle: list[list[tuple[End, End]]] = []
        for ti in range(t.n_triangles):
            stations = tri_segments[ti]
            # Build half-segment connectivity.  Identify each chord
            # segment by (chord, k); it joins station k and station k+1.
            # At a boundary station, the strand terminates; at a
            # crossing station, the strand continues to the half-edge
            # paired with the half-edge by which it arrived.
            #
            # Crossing g on chord c: arriving via segment on the 'in'
            # side means we came along half-edge (c,'in'); we continue
            # along pair[(c,'in')] = (c2, side2), i.e. onto the segment
            # of chord c2 adjacent to g on its side2.
            # Map (chord, station_index, direction) walks.
            # For chord c with stations st: segment k joins st[k], st[k+1].
            # At station st[k] = ('X', g): the segment on the 'in' side
            # is segment k-1...?  Convention: chord stations are ordered
            # from end0 to end1; for the crossing at station index k,
            # the 'in'-side segment is (c, k-1) and 'out'-side is (c, k).
            # Build adjacency: at ('X', g), for chord c at station k:
            #   half-edge (c,'in')  <-> segment (c, k-1)
            #   half-edge (c,'out') <-> segment (c, k)
            half_to_seg: dict[tuple[int, tuple[int, str]], tuple[int, int]] = {}
            seg_to_half: dict[tuple[int, int], list] = {}
            for c_idx, st in stations.items():
                for k, node in enumerate(st):
                    if node[0] != "X":
                        continue
                    g = node[1]
                    half_to_seg[(g, (c_idx, "in"))] = (c_idx, k - 1)
                    half_to_seg[(g, (c_idx, "out"))] = (c_idx, k)
            # boundary ends: segment (c, 0) starts at end0;
            # segment (c, len(st)-2) ends at end1.
            tri_arcs: list[tuple[End, End]] = []
            visited_segments: set[tuple[int, int]] = set()

            def walk_from_boundary(c_idx: int, at_start: bool):
                """Walk a strand starting at a boundary end."""
                st = stations[c_idx]
                if at_start:
                    seg = (c_idx, 0)
                    head_station = 1   # station index the segment leads to
                    start_end: End = st[0][1]
                    direction = +1
                else:
                    seg = (c_idx, len(st) - 2)
                    head_station = len(st) - 2
                    start_end = st[-1][1]
                    direction = -1
                cur_chord = c_idx
                cur_seg = seg
                cur_head = head_station
                cur_dir = direction
                while True:
                    visited_segments.add(cur_seg)
                    node = stations[cur_chord][cur_head]
                    if node[0] == "B":
                        return (start_end, node[1])
                    g = node[1]
                    # we arrive at crossing g along chord cur_chord; the
                    # half-edge by which we arrive: if we were moving in
                    # +1 direction, the segment is the 'in'-side of g on
                    # this chord, so we arrive along half-edge 'in'.
                    arrive = (cur_chord, "in" if cur_dir == +1 else "out")
                    depart = pair_at[g][arrive]
                    d_chord, d_side = depart
                    d_seg = half_to_seg[(g, (d_chord, d_side))]
                    # sanity: g's station on d_chord borders d_seg
                    k = d_seg[1] if d_side == "out" else d_seg[1] + 1
                    if stations[d_chord][k] != ("X", g):
                        raise AssertionError("station bookkeeping error")
                    # Departing along the 'out' side walks the chord in
                    # the +1 direction toward station d_seg[1] + 1;
                    # 'in' side walks -1 toward station d_seg[1].
                    if d_side == "out":
                        cur_chord, cur_seg, cur_dir = d_chord, d_seg, +1
                        cur_head = d_seg[1] + 1
                    else:
                        cur_chord, cur_seg, cur_dir = d_chord, d_seg, -1
                        cur_head = d_seg[1]

            # launch walks from all boundary ends
            done_ends: set[End] = set()
            for c_idx, st in stations.items():
                for at_start in (True, False):
                    end = st[0][1] if at_start else st[-1][1]
                    if end in done_ends:
                        continue
                    seg0 = (c_idx, 0) if at_start else (c_idx, len(st) - 2)
                    if seg0 in visited_segments:
                        continue
                    e0, e1 = walk_from_boundary(c_idx, at_start)
                    tri_arcs.append((e0, e1))
                    done_ends.add(e0)
                    done_ends.add(e1)
            # closed loops: any unvisited segments form cycles
            all_segs = {
                (c_idx, k)
                for c_idx, st in stations.items()
                for k in range(len(st) - 1)
            }
            remaining = all_segs - visited_segments
            while remaining:
                # walk a cycle
                c_idx, k = sorted(remaining)[0]
                cur_chord, cur_seg, cur_dir = c_idx, (c_idx, k), +1
                cur_head = k + 1
                while True:
                    remaining.discard(cur_seg)
                    visited_segments.add(cur_seg)
                    node = stations[cur_chord][cur_head]
                    if node[0] == "B":
                        raise AssertionError("open strand found in loop walk")
                    g = node[1]
                    arrive = (cur_chord, "in" if cur_dir == +1 else "out")
                    depart = pair_at[g][arrive]
                    d_chord, d_side = depart
                    d_seg = half_to_seg[(g, (d_chord, d_side))]
                    if d_side == "out":
                        cur_chord, cur_seg, cur_dir = d_chord, d_seg, +1
                        cur_head = d_seg[1] + 1
                    else:
                        cur_chord, cur_seg, cur_dir = d_chord, d_seg, -1
                        cur_head = d_seg[1]
                    if cur_seg not in remaining:
                        break
                n_loops += 1
            arcs_per_triangle.append(tri_arcs)

        diagram = MatchingDiagram(t, joint_points, arcs_per_triangle)
        coords, n_triv = diagram.normalize()
        results.append(ResolvedTerm(
            coords=coords, n_trivial=n_loops + n_triv, a_exponent=a_exp,
        ))

    return results
