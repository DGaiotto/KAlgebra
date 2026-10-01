"""Shared disk-resolution machinery for stated skein algebras of
polygons (disk primitives): the triangle (`stated_triangle`), the bigon
(`stated_bigon`), and any future n-gon primitive.

Given the joint boundary layout of two stacked diagrams in a disk —
boundary points 0..n-1 in ccw order, each assigned a boundary edge and
a state, the upper diagram's arcs and the lower diagram's arcs as chord
pairs — `resolve_chords` finds all crossings (exact rational chord
layout), resolves every Kauffman smoothing (upper strand = first
factor), evaluates trivial loops, and normalizes against the boundary:
innermost caps (C^+_- = q^{-1/2}, C^-_+ = -q^{-5/2}, like-states = 0)
and the height-exchange state sorting

    (bad pair: later -, earlier +) = q^{E_SWAP} (swapped)
                                   + q^{E_JOIN} (ends joined),

iterated to increasing states (terminating: inversions + endpoint count
strictly decrease).  The caller supplies `readoff(arcs, states,
edge_of)` mapping a normalized crossingless cap-free configuration to
its basis label.

The convention constants (cap values, exchange exponents, stacking
order, smoothing orientation) are the ones pinned EMPIRICALLY by the
triangle presentation rel1-rel4 (the suite in the source repository); they
live here as the single source of truth and are re-exported by
`stated_triangle` for backward compatibility.
"""

from __future__ import annotations

from fractions import Fraction

from half_laurent import HalfLaurent, ONE as HL_ONE
from skein_resolve import _angle_key, _seg_intersection

# Trivial loop value: -q^2 - q^{-2}
DELTA = HalfLaurent({2: -1, -2: -1})

# Cap values C^{eps}_{eps'} -- indexed (state at the LATER boundary
# point, state at the EARLIER point) along the boundary orientation;
# pinned by the rel1-rel4 harness.
C_CAP = {
    (1, 1): HalfLaurent.zero(),
    (-1, -1): HalfLaurent.zero(),
    (1, -1): HalfLaurent.monomial(Fraction(-1, 2)),
    (-1, 1): HalfLaurent.monomial(Fraction(-5, 2), -1),
}

# Height-exchange (eq.order) rewrite exponents, pinned by the harness:
#   (later -, earlier +)  =  q^{E_SWAP} (later +, earlier -)
#                          + q^{E_JOIN} (the two strand-ends joined)
E_SWAP = 2
E_JOIN = Fraction(-1, 2)

# Stacking: if True, the first (upper) factor's points sit LATER along
# each boundary edge; if False, earlier.  Pinned by the harness.
UPPER_LATER = True
# Smoothing weight orientation: the "ccw" half-edge pairing carries
# q^{-1} here (pinned by the rel1-rel4 harness: the unique convention
# combo passing the full presentation).
SMOOTH_SIGN = -1


def resolve_chords(n_points, arcs_a, arcs_b, states, edge_of, readoff):
    """Resolve the stacked chord diagram and normalize.

    n_points  boundary points 0..n_points-1 in ccw order
    arcs_a    upper diagram's arcs: list of (pid, pid)
    arcs_b    lower diagram's arcs: list of (pid, pid)
    states    dict pid -> +/-1
    edge_of   dict pid -> boundary edge index
    readoff   (arcs: set[frozenset], states, edge_of) -> label

    Returns dict[label, HalfLaurent].
    """
    chords = [(1, p, q) for (p, q) in arcs_a] + \
             [(0, p, q) for (p, q) in arcs_b]
    return resolve_layered(n_points, chords, states, edge_of, readoff)


def resolve_layered(n_points, layered_arcs, states, edge_of, readoff):
    """General form: each arc carries an integer layer (height); at a
    crossing of two arcs the HIGHER layer goes over.  Same-layer arcs
    are assumed disjoint (crossings between them are skipped, matching
    the historical two-layer behaviour).

    layered_arcs  list of (layer, pid, pid)

    Returns dict[label, HalfLaurent].
    """
    n = n_points
    pos = {}
    for i in range(n):
        t = Fraction(i) + Fraction((i * i + 1), 9973 * (n + 7))
        pos[i] = (t, t * t)

    chords = [(lay, i, p, q) for i, (lay, p, q) in enumerate(layered_arcs)]
    crossings = []
    for x1 in range(len(chords)):
        for x2 in range(x1 + 1, len(chords)):
            w1, i1, p1, q1 = chords[x1]
            w2, i2, p2, q2 = chords[x2]
            if w1 == w2:
                continue
            hit = _seg_intersection(pos[p1], pos[q1], pos[p2], pos[q2])
            if hit is None:
                continue
            s, t, x = hit
            over, under = (x1, x2) if w1 > w2 else (x2, x1)
            s_over = s if w1 > w2 else t
            s_under = t if w1 > w2 else s
            crossings.append((over, under, s_over, s_under, x))

    n_cross = len(crossings)
    out: dict = {}

    # stations per chord: ordered crossing parameters
    chord_cross: dict[int, list] = {ci: [] for ci in range(len(chords))}
    for g, (ov, un, s_ov, s_un, x) in enumerate(crossings):
        chord_cross[ov].append((s_ov, g))
        chord_cross[un].append((s_un, g))
    for ci in chord_cross:
        chord_cross[ci].sort()

    # half-edge angular data per crossing
    cross_geo = []
    for g, (ov, un, s_ov, s_un, x) in enumerate(crossings):
        local = {}
        for role, ci in (("over", ov), ("under", un)):
            _w, _i, p, q = chords[ci]
            local[(ci, "in")] = (pos[p][0] - x[0], pos[p][1] - x[1])
            local[(ci, "out")] = (pos[q][0] - x[0], pos[q][1] - x[1])
        order = sorted(local, key=lambda h: _angle_key(local[h]))
        cross_geo.append((ov, un, order))

    for mask in range(1 << n_cross):
        coeff = HL_ONE
        pairing_at = {}
        for g, (ov, un, order) in enumerate(cross_geo):
            use_ccw = bool((mask >> g) & 1)
            coeff = coeff.shift(SMOOTH_SIGN if use_ccw else -SMOOTH_SIGN)
            m = {}
            for posn, h in enumerate(order):
                if h[0] != ov:
                    continue
                nxt = order[(posn + (1 if use_ccw else -1)) % 4]
                m[h] = nxt
                m[nxt] = h
            pairing_at[g] = m

        # walk: stations of chord ci: ["P"p, X g1, .., X gk, "P"q]
        def stations(ci):
            _w, _i, p, q = chords[ci]
            return [("P", p)] + [("X", g) for _s, g in chord_cross[ci]] + [("P", q)]

        visited = set()

        def step(ci, j, direction):
            """From segment j of chord ci moving `direction`, return
            (endpoint or None, next (ci, j, direction))."""
            st = stations(ci)
            node = st[j + 1] if direction > 0 else st[j]
            if node[0] == "P":
                return node[1], None
            g = node[1]
            arrive = (ci, "in" if direction > 0 else "out")
            dep = pairing_at[g][arrive]
            d_ci, d_side = dep
            kpos = next(ix for ix, (_s, gg) in enumerate(chord_cross[d_ci]) if gg == g)
            if d_side == "out":
                return None, (d_ci, kpos + 1, +1)
            return None, (d_ci, kpos, -1)

        new_arcs = []
        for ci in range(len(chords)):
            st = stations(ci)
            for (j0, d0, startp) in ((0, +1, st[0][1]), (len(st) - 2, -1, st[-1][1])):
                if (ci, j0, "seen") in visited:
                    continue
                # walk
                cur = (ci, j0, d0)
                ends = None
                while True:
                    cci, cj, cd = cur
                    visited.add((cci, cj, "seen"))
                    endp, nxt = step(cci, cj, cd)
                    if endp is not None:
                        ends = endp
                        break
                    cur = nxt
                new_arcs.append((startp, ends))
        # dedupe: each open strand found twice (once from each end)
        seen_pairs = set()
        arcs_final = []
        for p, q in new_arcs:
            key = frozenset((p, q))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            arcs_final.append((p, q))
        # loops = unvisited segments
        n_loops = 0
        rem = set()
        for ci in range(len(chords)):
            for j in range(len(stations(ci)) - 1):
                if (ci, j, "seen") not in visited:
                    rem.add((ci, j))
        while rem:
            ci, j = sorted(rem)[0]
            cur = (ci, j, +1)
            while True:
                cci, cj, cd = cur
                if (cci, cj) in rem:
                    rem.discard((cci, cj))
                visited.add((cci, cj, "seen"))
                _endp, nxt = step(cci, cj, cd)
                assert _endp is None, "open strand inside loop walk"
                cur = nxt
                if (cur[0], cur[1]) not in rem:
                    break
            n_loops += 1
        for _ in range(n_loops):
            coeff = coeff * DELTA

        leaf = norm_rec(
            {frozenset(a) for a in arcs_final}, dict(states), edge_of, readoff,
        )
        for lbl, c in leaf.items():
            s = out.get(lbl, HalfLaurent.zero()) + c * coeff
            if s.is_zero():
                out.pop(lbl, None)
            else:
                out[lbl] = s
    return out


def _scaled(d: dict, c: HalfLaurent) -> dict:
    return {lbl: v * c for lbl, v in d.items()}


def _merged(d1: dict, d2: dict) -> dict:
    out = dict(d1)
    for lbl, c in d2.items():
        s = out.get(lbl, HalfLaurent.zero()) + c
        if s.is_zero():
            out.pop(lbl, None)
        else:
            out[lbl] = s
    return out


def norm_rec(arcs, states, edge_of, readoff) -> dict:
    """Boundary normalization: caps, then state sorting, then readoff.

    Returns dict[label, HalfLaurent].
    """
    # 1. caps: innermost returning arc on one edge
    live = sorted(states)
    for arc in sorted(arcs, key=sorted):
        pq = sorted(arc)
        if len(pq) != 2:
            continue
        p, q = pq
        if edge_of[p] != edge_of[q]:
            continue
        between = [r for r in live if p < r < q]
        if between:
            continue
        cap = C_CAP[(states[q], states[p])]
        if cap.is_zero():
            return {}
        rest_arcs = {a for a in arcs if a != arc}
        rest_states = {r: s for r, s in states.items() if r not in (p, q)}
        return _scaled(norm_rec(rest_arcs, rest_states, edge_of, readoff), cap)

    # 2. state sorting on adjacent same-edge pairs
    for i in range(len(live) - 1):
        p, q = live[i], live[i + 1]
        if edge_of[p] != edge_of[q]:
            continue
        if states[q] == -1 and states[p] == 1:
            swapped = dict(states)
            swapped[p], swapped[q] = -1, 1
            term1 = _scaled(norm_rec(arcs, swapped, edge_of, readoff),
                            HalfLaurent.monomial(E_SWAP))
            arc_p = next(a for a in arcs if p in a)
            arc_q = next(a for a in arcs if q in a)
            rest = {a for a in arcs if a not in (arc_p, arc_q)}
            rest_states = {r: s for r, s in states.items() if r not in (p, q)}
            if arc_p == arc_q:
                term2 = _scaled(norm_rec(rest, rest_states, edge_of, readoff),
                                DELTA)
            else:
                (a1,) = [x for x in arc_p if x != p]
                (b1,) = [x for x in arc_q if x != q]
                rest = rest | {frozenset((a1, b1))}
                term2 = norm_rec(rest, rest_states, edge_of, readoff)
            return _merged(term1,
                           _scaled(term2, HalfLaurent.monomial(E_JOIN)))

    # 3. read off the basis label
    return {readoff(arcs, states, edge_of): HL_ONE}
