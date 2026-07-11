"""The stated skein algebra of the ideal bigon -- intrinsic engine,
and the quantum-group base case.

S^s(B) for the bigon B (disk minus 2 boundary points) with boundary
edges a, b:
basis = theta(k) -- k parallel copies of the arc alpha joining a and b
-- with INCREASING states on each edge.  Geometry:
going ccw, edge a carries points 0..k-1, edge b points k..2k-1; the
parallel (non-crossing) pairing joins a-position i to b-position
k-1-i.

Product = stack first factor ABOVE second, resolve and normalize via
the shared `stated_disk` engine (the conventions pinned by the
triangle rel1-rel4 harness).  The bigon presentation is

    re1: alpha(e,-) alpha(e',+) = q^2 alpha(e,+) alpha(e',-)
                                  - q^{5/2} C^e_{e'}
    re2: alpha(-,e) alpha(+,e') = q^2 alpha(+,e) alpha(-,e')
                                  - q^{5/2} C^e_{e'}

which, with a = alpha(+,+), b = alpha(+,-), c = alpha(-,+),
d = alpha(-,-), is EXACTLY the standard O_Q(SL2) presentation at
Q = q^2:

    ba = Q ab,  ca = Q ac,  db = Q bd,  dc = Q cd,  bc = cb,
    ad - Q^{-1} bc = 1,  da - Q bc = 1

-- i.e. S^s(B) ~= O_{q^2}(SL2) as algebras.  The Hopf
structure: `coproduct` = Le's splitting along a vertical mid-arc:
state sum over the cut points, with ONE total order O lifted
to the heights on BOTH copies of the cut edge; the copy whose boundary
direction disagrees with O inherits REVERSED heights, expressed in the
positive-order basis by pushing the reversal into a half-twist braid
resolved by the Kauffman engine (higher strand over).  Delta is
independent of O -- asserted in the harness -- and the
bialgebra axioms (Delta/eps algebra maps, coassociativity, counit
axioms) all pass exactly: the matrix-coalgebra structure of
O_{q^2}(SL2) emerges from the geometry.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping

from half_laurent import HalfLaurent, ONE as HL_ONE
from stated_disk import UPPER_LATER, resolve_chords, resolve_layered

# Label: (k, s_a, s_b) with s_a, s_b the state tuples along edges a, b
# in the boundary orientation.
Label = tuple


def _readoff(arcs, states, edge_of) -> Label:
    """Normalized crossingless cap-free configuration -> basis label."""
    k = 0
    for arc in arcs:
        p, q = sorted(arc)
        if edge_of[p] == edge_of[q]:
            raise AssertionError(f"returning arc survived normalization: {arc}")
        k += 1
    s_by_edge = {0: [], 1: []}
    for r in sorted(states):
        s_by_edge[edge_of[r]].append(states[r])
    return (k, tuple(s_by_edge[0]), tuple(s_by_edge[1]))


def _stack(la: Label, lb: Label) -> dict:
    """Product of two basis diagrams (la ABOVE lb) as dict[Label, HL]."""
    ka, sa_a, sa_b = la
    kb, sb_a, sb_b = lb

    points = []
    states = {}
    edge_of = {}
    ids = {}

    def add_edge(eidx, tag_pts):
        # tag_pts: ordered list of (which, local_pos, state) along the
        # edge -- earlier first
        for which, j, st in tag_pts:
            pid = len(points)
            points.append(pid)
            states[pid] = st
            edge_of[pid] = eidx
            ids[(which, eidx, j)] = pid

    for eidx, (sa, sb) in enumerate(((sa_a, sb_a), (sa_b, sb_b))):
        lower = [("B", j, sb[j]) for j in range(len(sb))]
        upper = [("A", j, sa[j]) for j in range(len(sa))]
        add_edge(eidx, lower + upper if UPPER_LATER else upper + lower)

    def arcs_of(which, k):
        # parallel family: a-position i <-> b-position k-1-i
        return [
            (ids[(which, 0, i)], ids[(which, 1, k - 1 - i)])
            for i in range(k)
        ]

    arcs_a = arcs_of("A", ka)
    arcs_b = arcs_of("B", kb)
    return resolve_chords(len(points), arcs_a, arcs_b, states, edge_of,
                          _readoff)


class StatedBigon:
    """S^s(B) elements as dict[Label, HalfLaurent]; product by stacking."""

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
        return cls({(0, (), ()): HL_ONE})

    @classmethod
    def basis(cls, k, s_a, s_b, coeff: HalfLaurent | int = 1):
        if isinstance(coeff, int):
            coeff = HalfLaurent.from_int(coeff)
        return cls({(k, tuple(s_a), tuple(s_b)): coeff})

    @classmethod
    def alpha(cls, eps, eps_p):
        """alpha(eps, eps'): the a<->b arc, eps = state on edge a,
        eps' = state on edge b (Le's convention)."""
        return cls.basis(1, (eps,), (eps_p,))

    # -- algebra ---------------------------------------------------------

    def __add__(self, o):
        out = dict(self.terms)
        for lbl, c in o.terms.items():
            s = out.get(lbl, HalfLaurent.zero()) + c
            if s.is_zero():
                out.pop(lbl, None)
            else:
                out[lbl] = s
        return StatedBigon(out)

    def __sub__(self, o):
        return self + (-1) * o

    def __mul__(self, o):
        if isinstance(o, int):
            o = HalfLaurent.from_int(o)
        if isinstance(o, HalfLaurent):
            return StatedBigon({l: c * o for l, c in self.terms.items()})
        out = StatedBigon.zero()
        for la, ca in self.terms.items():
            for lb, cb in o.terms.items():
                prod = StatedBigon(_stack(la, lb))
                out = out + prod * (ca * cb)
        return out

    __rmul__ = __mul__

    def __eq__(self, o):
        return isinstance(o, StatedBigon) and self.terms == o.terms

    def __repr__(self):
        if not self.terms:
            return "0"
        bits = []
        for lbl in sorted(self.terms):
            k, sa, sb = lbl
            bits.append(f"({self.terms[lbl]!r})*theta{k}[a:{sa} b:{sb}]")
        return " + ".join(bits)


# ---------------------------------------------------------------------------
# Hopf structure (S^s(B) ~= O_{q^2}(SL2))
# ---------------------------------------------------------------------------
#
# Delta = Le's splitting rho along a vertical mid-arc c: cut
# each strand, sum over states mu at the cut, and lift ONE total order
# O on the cut points to the heights on BOTH copies of c.  The two
# copies are traversed oppositely by the boundary orientations of the
# two halves, so whichever half has its positive direction disagree
# with O inherits REVERSED heights on its cut edge; expressing that
# half in the (positive-order) basis pushes the reversal into actual
# crossings -- a half-twist braid resolved by the Kauffman engine
# (strand with higher height goes over).  rho is independent of the
# choice of O -- verified empirically by computing both
# choices; the bialgebra harness pins everything else.
#
# Geometry of theta(k, s_a, s_b) (horizontal strands at levels
# 0..k-1, edge a on the left read upward, edge b on the right read
# downward): strand at level l meets a at position l, b at position
# k-1-l, and the mid-arc at level l.


def _half_normalize(k, edge0_states, edge1_states):
    """Expand the flat k-strand parallel diagram with the given states
    (by POSITION on each edge) in the basis -- pure cap/state-sort
    normalization through the engine."""
    states = {}
    edge_of = {}
    for j in range(k):
        states[j] = edge0_states[j]
        edge_of[j] = 0
        states[k + j] = edge1_states[j]
        edge_of[k + j] = 1
    chords = [(0, l, k + (k - 1 - l)) for l in range(k)]
    return resolve_layered(2 * k, chords, states, edge_of, _readoff)


def coproduct(x: StatedBigon, O_upward: bool = True):
    """Delta: S^s(B) -> S^s(B) (x) S^s(B), Le's splitting along the
    vertical mid-arc.  Returns dict[(Label, Label), HalfLaurent].

    O_upward chooses the lifted total order on the cut points; the
    result is independent of it -- asserted in the tests.
    """
    out: dict = {}
    for (k, s_a, s_b), c in x.terms.items():
        for m in range(1 << k):
            # mu[l] = state of the strand at level l
            mu = tuple(1 if (m >> l) & 1 else -1 for l in range(k))
            # LEFT half: edges (a, c_L); c_L read DOWNWARD (positive
            # direction of the left half): position j = level k-1-j.
            # heights = O: level l has O-rank l (upward) or k-1-l.
            left_cut_states = tuple(mu[k - 1 - j] for j in range(k))
            # Heights on the cut edge: positive order on c_L means
            # heights increasing along c_L (downward, i.e. decreasing
            # level).  O upward gives heights increasing with level:
            # REVERSED on c_L (twist); O downward: flat.
            if O_upward:
                left = _twist_expand(k, s_a, mu, side="left")
            else:
                left = {(k, s_a, left_cut_states): HL_ONE}
                left = _sort_only(left)
            # RIGHT half: edges (c_R, b); c_R read UPWARD: position
            # j = level j; O upward = positive order, O downward =
            # reversed.
            right_cut_states = tuple(mu[j] for j in range(k))
            if O_upward:
                right = {(k, right_cut_states, s_b): HL_ONE}
                right = _sort_only(right)
            else:
                right = _twist_expand(k, s_b, mu, side="right")
            for ll, cl in left.items():
                for rr, cr in right.items():
                    key = (ll, rr)
                    s = out.get(key, HalfLaurent.zero()) + c * cl * cr
                    if s.is_zero():
                        out.pop(key, None)
                    else:
                        out[key] = s
    return out


def _sort_only(terms: dict) -> dict:
    """Normalize raw labels (possibly non-increasing states) into the
    basis: resolve the flat parallel diagram through the engine."""
    out: dict = {}
    for (k, s0, s1), c in terms.items():
        for lbl, cc in _half_normalize(k, s0, s1).items():
            s = out.get(lbl, HalfLaurent.zero()) + c * cc
            if s.is_zero():
                out.pop(lbl, None)
            else:
                out[lbl] = s
    return out


def _twist_expand(k, far_states, mu, side: str) -> dict:
    """The half whose cut edge inherits REVERSED heights: re-attach
    the cut endpoints in height order (half-twist braid; higher height
    over) and resolve.

    side="left":  edges (a=edge0 with far_states, c_L=edge1); strand
        level l at a-pos l, c_L-pos k-1-l, height rank l (O upward).
    side="right": edges (c_R=edge0, b=edge1 with far_states); strand
        level l at c_R-pos l, b-pos k-1-l, height rank k-1-l
        (O downward).
    Both cases: cut endpoint of strand l slides from its position to
    the position dictated by its height; states ride along.
    """
    states = {}
    edge_of = {}
    if side == "left":
        # edge0 = a (far), edge1 = c_L (cut, reversed heights)
        for j in range(k):
            states[j] = far_states[j]
            edge_of[j] = 0
            edge_of[k + j] = 1
        # strand l: a-pos l; cut height rank l; positive order on c_L
        # wants heights increasing with c_L-position, so strand l goes
        # to c_L-position l (instead of k-1-l): chords (l, k+l) -- the
        # crossing pattern -- with layer = height = l; cut state mu[l]
        # sits at c_L-position l.
        for l in range(k):
            states[k + l] = mu[l]
        chords = [(l, l, k + l) for l in range(k)]
    else:
        # edge0 = c_R (cut, reversed heights), edge1 = b (far)
        for j in range(k):
            states[k + j] = far_states[j]
            edge_of[k + j] = 1
            edge_of[j] = 0
        # strand l: b-pos k-1-l; cut height rank k-1-l (O downward);
        # positive order on c_R wants strand l at c_R-position k-1-l:
        # chords (k-1-l, k + (k-1-l)) with layer = k-1-l; cut state
        # mu[l] at c_R-position k-1-l.
        for l in range(k):
            states[k - 1 - l] = mu[l]
        chords = [(k - 1 - l, k - 1 - l, k + (k - 1 - l)) for l in range(k)]
    return resolve_layered(2 * k, chords, states, edge_of, _readoff)


def counit(x: StatedBigon) -> HalfLaurent:
    """epsilon: S^s(B) -> R, the algebra map with
    eps(alpha(e,e')) = delta_{e,e'} (the O_q(SL2) counit).  On the
    basis: theta(k) with constant equal states on both edges maps to
    q^{k(k-1)/2} (the flat k-strand diagram re-expanded through the
    stacked generators), all other basis elements to 0.  Validated by
    the counit axioms in the harness."""
    tot = HalfLaurent.zero()
    for (k, s_a, s_b), c in x.terms.items():
        if (len(set(s_a)) <= 1 and s_a == s_b):
            tot = tot + c * HalfLaurent.monomial(
                Fraction(k * (k - 1), 2) if k else 0)
    return tot


def tensor_mul(x: dict, y: dict) -> dict:
    """Componentwise product on S^s(B) (x) S^s(B) tensors."""
    out: dict = {}
    for (l1, r1), c1 in x.items():
        for (l2, r2), c2 in y.items():
            left = StatedBigon({l1: HL_ONE}) * StatedBigon({l2: HL_ONE})
            right = StatedBigon({r1: HL_ONE}) * StatedBigon({r2: HL_ONE})
            for ll, cl in left.terms.items():
                for rr, cr in right.terms.items():
                    key = (ll, rr)
                    s = out.get(key, HalfLaurent.zero()) + c1 * c2 * cl * cr
                    if s.is_zero():
                        out.pop(key, None)
                    else:
                        out[key] = s
    return out
