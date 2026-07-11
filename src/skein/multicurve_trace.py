"""EXPERIMENTAL multi-arc state sum -- known to be incomplete.

Status.  The naive extension of the single-arc state sum
(product of per-arc `phi` values in a fixed per-triangle order) is
**not correct in general** for multi-arc inputs, even after the
projection reorder fix in `y_delta.project_triangle_monomial`:

* it fails parallel-copy multiplicativity
  (`state_sum(2*gamma) != Tr(gamma)^2`) -- for multi-COMPONENT
  diagrams the relative stacking of components is genuine data (the
  basis diagram stacks the components, i.e. the disjoint union maps to
  the PRODUCT of component traces), which a flat per-triangle order
  does not realize;
* for connected multi-arc curves it matches the overdetermined
  product-transport answer on the symmetric twisted curve
  (1,1,2,2,1,1) of the tetrahedron but NOT on its images
  (1,2,1,1,2,1), (2,1,1,1,1,2): when one strand crosses the same edge
  twice, the heights at those two points are coupled along the strand,
  and the positive-order corrections contribute q-powers and
  state-swap terms this module does not implement.

Implementing the full elevation calculus is open work.  The correct
production route for
arbitrary multicurves is `geometric_f.GeometricF`: traces of connected
curves by constructive transport along intrinsic skein products
(every entry overdetermined and cross-checked), disjoint unions as
products of component traces.

Single-arc inputs agree with `quantum_trace.quantum_trace`; calls with
a genuinely multi-arc input raise unless `experimental=True`.
"""

from __future__ import annotations

from typing import Sequence

from half_laurent import HalfLaurent, ONE as HL_ONE
from multicurve import is_admissible, normal_arcs
from phi_map import ALPHA, BETA, GAMMA, phi
from triangle_algebra import TriangleAlgebra
from triangulation import Triangulation
from y_delta import YDelta, project_triangle_monomial

_ARC_TYPE_BY_CORNER = {0: ALPHA, 1: BETA, 2: GAMMA}
# corner k joins ccw positions (k+1) [earlier, state eps'] and
# (k+2) [later, state eps]; the existing phi conventions assign
# {1,2} -> alpha, {2,0} -> beta, {0,1} -> gamma, matching this table.


def multicurve_trace(
    triangulation: Triangulation,
    coords: Sequence[int],
    *,
    arc_order: str = "corner_depth",
    experimental: bool = False,
) -> YDelta:
    """Naive multi-arc state sum -- see module docstring (EXPERIMENTAL).

    Agrees with `quantum_trace` on single-arc inputs; for genuinely
    multi-arc inputs it is known to be missing the boundary-reorder
    corrections and raises unless `experimental=True`.
    """
    t = triangulation
    coords = tuple(coords)
    if not is_admissible(t, coords):
        raise ValueError(f"{coords} is not an admissible multicurve")
    arcs = normal_arcs(t, coords)
    if any(len(tri) > 1 for tri in arcs) and not experimental:
        raise NotImplementedError(
            "naive multi-arc state sum is incomplete (boundary-reorder "
            "corrections missing); use geometric_f.GeometricF, or pass "
            "experimental=True to evaluate the naive formula anyway"
        )

    # State index space: one sign per intersection point (edge, point).
    points: list[tuple[int, int]] = []
    point_idx: dict[tuple[int, int], int] = {}
    for e in range(t.n_edges):
        for p in range(coords[e]):
            point_idx[(e, p)] = len(points)
            points.append((e, p))
    n_pts = len(points)

    # Pre-sort the arcs of each triangle in the chosen order.
    def _key(a):
        if arc_order == "corner_depth":
            return (a.corner_pos, a.depth)
        elif arc_order == "depth_corner":
            return (-a.depth, -a.corner_pos)
        raise ValueError(f"unknown arc_order {arc_order!r}")

    sorted_arcs = [sorted(tri, key=_key) for tri in arcs]

    result = YDelta.zero(t)

    for state_int in range(1 << n_pts):
        sign = [1 if (state_int >> i) & 1 else -1 for i in range(n_pts)]

        triangle_exps: list[tuple[int, int, int]] = []
        global_coef: HalfLaurent = HL_ONE
        annihilated = False
        for ti in range(t.n_triangles):
            factor = TriangleAlgebra.one()
            for arc in sorted_arcs[ti]:
                eps = sign[point_idx[arc.end_b]]        # ccw-later endpoint
                eps_prime = sign[point_idx[arc.end_a]]  # ccw-earlier
                f = phi(_ARC_TYPE_BY_CORNER[arc.corner_pos], eps, eps_prime)
                if f.is_zero():
                    annihilated = True
                    break
                factor = factor * f
            if annihilated:
                break
            terms = factor.items()
            if len(terms) != 1:
                raise AssertionError("triangle factor not a monomial")
            (exp_triple, c_local), = terms
            triangle_exps.append(exp_triple)
            global_coef = global_coef * c_local
        if annihilated:
            continue

        proj = project_triangle_monomial(t, triangle_exps, global_coef)
        if proj is None:
            continue
        charge_vec, ordered_coef = proj
        result = result + YDelta.monomial(t, charge_vec, ordered_coef)

    return result
