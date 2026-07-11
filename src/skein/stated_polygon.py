"""Stated skein algebra of a triangulated polygon, by triangular
decomposition.

Le's splitting homomorphism rho: S^s(P) -> (x)_T S^s(T) along the
internal edges is an injective ALGEBRA map to the ordinary tensor
product; we therefore work
directly inside the tensor algebra:

  * elements  = dict[tuple of per-triangle labels, HalfLaurent];
  * product   = componentwise `stated_triangle` products (the pinned
    triangle engine), distributed;
  * geometric boundary arcs = their rho-images: route the arc through
    the dual tree of the triangulation (unique for polygons), one
    corner-arc piece per visited triangle, and sum over +-1 states at
    each internal-edge crossing -- pure state sum, no q-factors
    (single crossing per edge, so no height-order subtlety; the
    multi-strand braid corrections of `stated_bigon.coproduct` arise
    only for parallel families, which here appear via products, whose
    heights the per-triangle engine already tracks).

Validation: the n=3 polygon
reproduces the triangle generators verbatim; routed arcs in the square and
pentagon satisfy Le's bigon presentation re1/re2 (cap normalization
through the gluing); the universal height-exchange relation
X(-)Y(+) = q^2 X(+)Y(-) + r * join holds with the SAME r for n=3
in-triangle configurations and n=4/5 routed configurations of the
same ccw orientation class (routing invariance of the local boundary
algebra); disjoint arcs commute.
"""

from __future__ import annotations

from itertools import product as iproduct
from typing import Mapping

from bordered_triangulation import (
    BorderedTriangulation, ROLE_A, ROLE_B, ROLE_C,
)
from half_laurent import HalfLaurent, ONE as HL_ONE
from stated_triangle import StatedTriangle

TriLabel = tuple
ID_TRI: TriLabel = ((0, 0, 0), (), (), ())

# corner-arc k-triples by the pair of roles the arc joins
_KT = {
    frozenset((ROLE_B, ROLE_C)): (1, 0, 0),   # alpha
    frozenset((ROLE_C, ROLE_A)): (0, 1, 0),   # beta
    frozenset((ROLE_A, ROLE_B)): (0, 0, 1),   # gamma
}


def _corner_label(role1: int, s1: int, role2: int, s2: int) -> TriLabel:
    """Single corner arc joining role1 (state s1) and role2 (state s2)."""
    kt = _KT[frozenset((role1, role2))]
    states = {role1: (s1,), role2: (s2,)}
    return (kt, states.get(0, ()), states.get(1, ()), states.get(2, ()))


class StatedPolygon:
    """S^s of a tree-dual triangulated polygon, inside (x)_T S^s(T).

    Construct with an int `n` (the fan-triangulated n-gon, the
    original scope) or with any `BorderedTriangulation` whose dual
    graph is a connected TREE (a triangulated disk — fan, zigzag, or
    general; the tree guarantees unique arc routing).
    """

    def __init__(self, n):
        if isinstance(n, BorderedTriangulation):
            self.t = n
            self.n = self.t.n_boundary
        else:
            self.n = n
            self.t = BorderedTriangulation.fan_polygon(n)
        self.m = self.t.n_triangles
        if len(self.t.gluings) != self.m - 1:
            raise NotImplementedError(
                "the stated engine needs a tree-dual (disk) "
                "triangulation: #gluings must be #triangles - 1")
        seen = {0}
        stack = [0]
        while stack:
            for _r, t2, _r2 in self.t.dual_neighbours(stack.pop()):
                if t2 not in seen:
                    seen.add(t2)
                    stack.append(t2)
        if len(seen) != self.m:
            raise NotImplementedError("dual graph is not connected")
        self._mult_cache: dict = {}

    # -- elements --------------------------------------------------------

    def zero(self) -> "PolyElement":
        return PolyElement(self, {})

    def one(self) -> "PolyElement":
        return PolyElement(self, {(ID_TRI,) * self.m: HL_ONE})

    def arc(self, s: int, t: int, eps_s: int, eps_t: int) -> "PolyElement":
        """The embedded arc joining polygon sides s and t, with states
        eps_s, eps_t -- as its rho-image (state sum over the internal
        edges it crosses)."""
        if s == t:
            raise ValueError("returning arcs are not generators here")
        slot_s = self.t.side_of_polygon(s)
        slot_t = self.t.side_of_polygon(t)
        chain = self.t.dual_path(slot_s[0], slot_t[0])
        # chain: [(tri, in_role, out_role)] with first in_role None /
        # last out_role None; replace the Nones by the boundary roles
        chain = [
            (tri,
             slot_s[1] if i == 0 else in_r,
             slot_t[1] if i == len(chain) - 1 else out_r)
            for i, (tri, in_r, out_r) in enumerate(chain)
        ]
        if len(chain) == 1:
            tri, r1, r2 = chain[0]
            lab = list((ID_TRI,) * self.m)
            lab[tri] = _corner_label(r1, eps_s, r2, eps_t)
            return PolyElement(self, {tuple(lab): HL_ONE})
        n_cuts = len(chain) - 1
        out: dict = {}
        for m in range(1 << n_cuts):
            mu = [1 if (m >> i) & 1 else -1 for i in range(n_cuts)]
            lab = list((ID_TRI,) * self.m)
            for i, (tri, r_in, r_out) in enumerate(chain):
                x_in = eps_s if i == 0 else mu[i - 1]
                x_out = eps_t if i == len(chain) - 1 else mu[i]
                lab[tri] = _corner_label(r_in, x_in, r_out, x_out)
            key = tuple(lab)
            cur = out.get(key, HalfLaurent.zero()) + HL_ONE
            out[key] = cur
        return PolyElement(self, out)

    # -- product ---------------------------------------------------------

    def _mul_component(self, la: TriLabel, lb: TriLabel) -> dict:
        key = (la, lb)
        if key not in self._mult_cache:
            prod = StatedTriangle({la: HL_ONE}) * StatedTriangle({lb: HL_ONE})
            self._mult_cache[key] = prod.terms
        return self._mult_cache[key]

    def multiply(self, x: "PolyElement", y: "PolyElement") -> "PolyElement":
        out: dict = {}
        for ka, ca in x.terms.items():
            for kb, cb in y.terms.items():
                comp = [self._mul_component(ka[i], kb[i])
                        for i in range(self.m)]
                for combo in iproduct(*(c.items() for c in comp)):
                    lab = tuple(l for l, _c in combo)
                    coeff = ca * cb
                    for _l, c in combo:
                        coeff = coeff * c
                    s = out.get(lab, HalfLaurent.zero()) + coeff
                    if s.is_zero():
                        out.pop(lab, None)
                    else:
                        out[lab] = s
        return PolyElement(self, out)


class PolyElement:
    """An element of (x)_T S^s(T) over a StatedPolygon."""

    __slots__ = ("algebra", "terms")

    def __init__(self, algebra: StatedPolygon,
                 terms: Mapping[tuple, HalfLaurent] | None = None):
        self.algebra = algebra
        clean: dict = {}
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

    def __add__(self, o: "PolyElement") -> "PolyElement":
        out = dict(self.terms)
        for lbl, c in o.terms.items():
            s = out.get(lbl, HalfLaurent.zero()) + c
            if s.is_zero():
                out.pop(lbl, None)
            else:
                out[lbl] = s
        return PolyElement(self.algebra, out)

    def __sub__(self, o: "PolyElement") -> "PolyElement":
        return self + (-1) * o

    def __mul__(self, o):
        if isinstance(o, int):
            o = HalfLaurent.from_int(o)
        if isinstance(o, HalfLaurent):
            return PolyElement(self.algebra,
                               {l: c * o for l, c in self.terms.items()})
        if isinstance(o, PolyElement):
            return self.algebra.multiply(self, o)
        return NotImplemented

    __rmul__ = __mul__

    def __eq__(self, o):
        return isinstance(o, PolyElement) and self.terms == o.terms

    def is_zero(self) -> bool:
        return not self.terms

    def __repr__(self):
        if not self.terms:
            return "0"
        bits = []
        for lbl in sorted(self.terms):
            bits.append(f"({self.terms[lbl]!r})*{lbl}")
        return " + ".join(bits)
