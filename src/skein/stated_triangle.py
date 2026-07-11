"""The stated skein algebra of the ideal triangle -- intrinsic engine.

S^s(T) for the triangle T with boundary edges a, b, c (ccw): stated
tangle diagrams modulo the
Kauffman relation (crossing = q (I) + q^{-1} (II)), the trivial loop
(-q^2 - q^{-2}), the boundary cap relations and the
boundary height-exchange relation.

Basis (specialized to the triangle): flat simple
diagrams theta(k1, k2, k3) -- k1 alpha-arcs (joining b, c), k2
beta-arcs (c, a), k3 gamma-arcs (a, b), corner-nested -- with
INCREASING states on each boundary edge (no - above a +).  In a disk,
crossingless + cap-free forces the nested corner form, so the engine's
normalization provably lands in this basis.

The product stacks the first factor ABOVE the second, finds crossings
by the exact-rational chord layout (shared with `skein_resolve`),
resolves all Kauffman smoothings, then normalizes: trivial loops,
innermost caps (C^+_- = q^{-1/2}, C^-_+ = -q^{-5/2}, C^+_+ = C^-_- = 0),
and state-sorting via the height-exchange rewrite

    (bad pair: higher -, lower +)  =  q^{e_swap} (swapped states)
                                    + q^{e_join} (the two ends joined),

iterated to the increasing basis (terminating: inversions + endpoint
count strictly decrease).  The orientation conventions (which endpoint
is "later", the cap index order, the exchange exponents e_swap/e_join,
the stacking block order) are pinned EMPIRICALLY by requiring Le's
presentation rel1-rel4 (+ tau-rotations) to hold; the pinned values
are the defaults.

This is the polygon building block of the bordered-surface program:
polygons glue from
triangles; cutting/gluing along edges is Le's splitting, matched on
charts by directional node-drop RG flows.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

from half_laurent import HalfLaurent, ONE as HL_ONE
# Convention constants + the chord-resolution/normalization engine live
# in `stated_disk` (shared with the bigon and future polygon
# primitives); re-exported here for backward compatibility.
from stated_disk import (
    C_CAP, DELTA, E_JOIN, E_SWAP, SMOOTH_SIGN, UPPER_LATER,
    resolve_chords,
)


# Labels: (k, s_a, s_b, s_c) with k = (k1, k2, k3) the corner-arc
# counts and s_e the state tuple along edge e in the boundary
# orientation.  Edge contents (in order along each edge):
#   a: k2 beta-ends (outermost..innermost at the a^c corner first),
#      then k3 gamma-ends;  b: k3 gamma-ends, then k1 alpha-ends;
#   c: k1 alpha-ends, then k2 beta-ends.
# The nesting pairs the j-th point of each corner block with the
# matching j-th point on the adjacent edge (innermost = closest to the
# shared corner).
Label = tuple


def _edge_points(k):
    """Per edge, the list of (arc_type, depth) along the edge order.

    Arc types 0 = alpha (b<->c), 1 = beta (c<->a), 2 = gamma (a<->b).
    Depth 0 = innermost at the corner where the block sits.

    Geometry (triangle vertices v0, v1, v2 ccw; edge a = v0->v1,
    b = v1->v2, c = v2->v0): the beta-block on a sits at the v0 end
    (corner with c), the gamma-block at the v1 end (corner with b),
    etc.  Along each edge's direction the first block's depths DECREASE
    toward the corner shared with the previous edge... we fix the
    convention: on edge a: beta-ends with depth descending (k2-1 .. 0
    is wrong -- innermost nearest the corner at v0 means FIRST along a
    is the outermost beta): order = beta depths (k2-1, .., 0) then
    gamma depths (0, .., k3-1).  Cyclically for b and c.
    """
    k1, k2, k3 = k
    a = [(1, d) for d in range(k2 - 1, -1, -1)] + [(2, d) for d in range(k3)]
    b = [(2, d) for d in range(k3 - 1, -1, -1)] + [(0, d) for d in range(k1)]
    c = [(0, d) for d in range(k1 - 1, -1, -1)] + [(1, d) for d in range(k2)]
    return a, b, c


class StatedTriangle:
    """S^s(T) elements as dict[Label, HalfLaurent]; product by stacking."""

    def __init__(self, terms: Mapping[Label, HalfLaurent] | None = None):
        clean: dict[Label, HalfLaurent] = {}
        if terms:
            for lbl, c in terms.items():
                if c.is_zero():
                    continue
                cur = clean.get(lbl)
                s = c if cur is None else cur + c
                if s.is_zero():
                    clean.pop(lbl, None)
                else:
                    clean[lbl] = s
        self.terms = clean

    # -- constructors ---------------------------------------------------

    @classmethod
    def zero(cls):
        return cls()

    @classmethod
    def one(cls):
        return cls({((0, 0, 0), (), (), ()): HL_ONE})

    @classmethod
    def basis(cls, k, s_a, s_b, s_c, coeff: HalfLaurent | int = 1):
        if isinstance(coeff, int):
            coeff = HalfLaurent.from_int(coeff)
        return cls({(tuple(k), tuple(s_a), tuple(s_b), tuple(s_c)): coeff})

    @classmethod
    def alpha(cls, eps, eps_p):
        """alpha(eps, eps'): the b<->c arc, eps = state at alpha cap c,
        eps' = state at alpha cap b (Le's convention)."""
        return cls.basis((1, 0, 0), (), (eps_p,), (eps,))

    @classmethod
    def beta(cls, eps, eps_p):
        """beta = tau(alpha): the c<->a arc, eps at a, eps' at c."""
        return cls.basis((0, 1, 0), (eps,), (), (eps_p,))

    @classmethod
    def gamma(cls, eps, eps_p):
        """gamma = tau^2(alpha): the a<->b arc, eps at b, eps' at a."""
        return cls.basis((0, 0, 1), (eps_p,), (eps,), ())

    # -- algebra ---------------------------------------------------------

    def __add__(self, o):
        out = dict(self.terms)
        for lbl, c in o.terms.items():
            s = out.get(lbl, HalfLaurent.zero()) + c
            if s.is_zero():
                out.pop(lbl, None)
            else:
                out[lbl] = s
        return StatedTriangle(out)

    def __sub__(self, o):
        return self + (-1) * o

    def __mul__(self, o):
        if isinstance(o, int):
            o = HalfLaurent.from_int(o)
        if isinstance(o, HalfLaurent):
            return StatedTriangle({l: c * o for l, c in self.terms.items()})
        out = StatedTriangle.zero()
        for la, ca in self.terms.items():
            for lb, cb in o.terms.items():
                prod = _stack(la, lb)
                out = out + prod * (ca * cb)
        return out

    __rmul__ = __mul__

    def __eq__(self, o):
        return isinstance(o, StatedTriangle) and self.terms == o.terms

    def __repr__(self):
        if not self.terms:
            return "0"
        bits = []
        for lbl in sorted(self.terms):
            k, sa, sb, sc = lbl
            bits.append(f"({self.terms[lbl]!r})*theta{k}[a:{sa} b:{sb} c:{sc}]")
        return " + ".join(bits)


# ---------------------------------------------------------------------------
# The stacking product on basis diagrams
# ---------------------------------------------------------------------------


def _layout(la: Label, lb: Label):
    """Joint boundary layout: diagram A stacked ABOVE diagram B.

    Returns (points, arcs_a, arcs_b, states): `points` is the ccw list
    of joint boundary point ids; arcs are lists of (pid, pid) pairs;
    states maps pid -> +/-1.  On each edge, B's points come first
    (lower heights = earlier along the boundary orientation), then A's.
    """
    ids = {}
    points = []
    states = {}

    def add_edge(eidx, lb_pts, la_pts, lb_states, la_states, tag):
        seq = []
        order = (("B", lb_pts, lb_states), ("A", la_pts, la_states)) if UPPER_LATER \
            else (("A", la_pts, la_states), ("B", lb_pts, lb_states))
        for which, pts, sts in order:
            for j, (atype, depth) in enumerate(pts):
                pid = len(points)
                points.append(pid)
                ids[(which, tag, j)] = pid
                states[pid] = sts[j]
                seq.append(pid)
        return seq

    out_pts = []
    for tag, eidx in (("a", 0), ("b", 1), ("c", 2)):
        ka, kb = la[0], lb[0]
        pa = _edge_points(ka)[eidx]
        pb = _edge_points(kb)[eidx]
        sa = la[1 + eidx]
        sb = lb[1 + eidx]
        out_pts += add_edge(eidx, pb, pa, sb, sa, tag)

    def arcs_of(which, k):
        k1, k2, k3 = k
        ea, eb, ec = _edge_points(k)
        arcs = []
        # pair matching (arc_type, depth) endpoints across edges
        def pid_of(tag, plist, atype, depth):
            j = plist.index((atype, depth))
            return ids[(which, tag, j)]
        for d in range(k1):
            arcs.append((pid_of("b", eb, 0, d), pid_of("c", ec, 0, d)))
        for d in range(k2):
            arcs.append((pid_of("c", ec, 1, d), pid_of("a", ea, 1, d)))
        for d in range(k3):
            arcs.append((pid_of("a", ea, 2, d), pid_of("b", eb, 2, d)))
        return arcs

    arcs_a = arcs_of("A", la[0])
    arcs_b = arcs_of("B", lb[0])
    return points, arcs_a, arcs_b, states


def _readoff(arcs, states, edge_of) -> Label:
    """Normalized crossingless cap-free configuration -> basis label."""
    k = [0, 0, 0]
    for arc in arcs:
        p, q = sorted(arc)
        pair = frozenset((edge_of[p], edge_of[q]))
        if pair == frozenset((1, 2)):
            k[0] += 1
        elif pair == frozenset((2, 0)):
            k[1] += 1
        elif pair == frozenset((0, 1)):
            k[2] += 1
        else:
            raise AssertionError(f"unclassified arc {arc}")
    s_by_edge = {0: [], 1: [], 2: []}
    for r in sorted(states):
        s_by_edge[edge_of[r]].append(states[r])
    return (
        tuple(k), tuple(s_by_edge[0]), tuple(s_by_edge[1]), tuple(s_by_edge[2]),
    )


def _stack(la: Label, lb: Label) -> StatedTriangle:
    """Product of two basis diagrams: resolve crossings, normalize."""
    points, arcs_a, arcs_b, states = _layout(la, lb)
    n = len(points)
    ka, kb = la[0], lb[0]
    # edge of each joint point, in layout order: per edge tag the block
    # B-then-A; reconstruct lengths from the corner data
    edge_of = {}
    idx = 0
    for eidx in range(3):
        nb = len(_edge_points(kb)[eidx])
        na = len(_edge_points(ka)[eidx])
        for _ in range(nb + na):
            edge_of[idx] = eidx
            idx += 1
    assert idx == n

    return StatedTriangle(
        resolve_chords(n, arcs_a, arcs_b, states, edge_of, _readoff)
    )
