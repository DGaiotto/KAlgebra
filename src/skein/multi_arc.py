"""`multi_arc` — the general MULTI-ARC-PER-TRIANGLE stated quantum-trace
primitive.

`cut_reglue.CutReglue.arc_image` realizes a stated curve that crosses each
triangle at most once: per triangle a single arc, one `phi`-monomial, then
`project_triangle_monomial` + Weyl-correction + internal state-sum.  That
single-arc restriction (`curve_realization.realize_curve`'s
`sum(m) > 1` honest-fail) is exactly what blocks the SU(2)+N_f charts,
whose corner units (e.g. `s_A = (1,0,1|2,0,0)`) genuinely cross one
triangle twice.

This module lifts that restriction, verbatim, to any
`BorderedTriangulation` / `Triangulation`:

  * :func:`multi_local_realize` — the `sum(m) >= 2` generalization of
    `realize_curve`: per triangle the FULL list of `m_k` corner-type-`k`
    arc descriptors, never honest-failing on a multi-arc triangle.

  * :func:`multi_arc_ordered` — the certified core.  The ONLY new
    ingredient over `arc_image` is that, per triangle, the arcs'
    `phi`-monomials are multiplied inside that triangle's
    :class:`TriangleAlgebra` in a chosen ELEVATION order
    (`order_by_tri[ti]`; reversed elevation, since `phi` is an
    anti-homomorphism); everything else — the internal-edge state sum,
    `project_triangle_monomial`, the coefficient bookkeeping — is
    identical to `arc_image`.  Returns the RAW ordered-form
    (`charge -> HalfLaurent`) dict, the frame `skein_a1d3_kalg.skein_mul`
    lives in; corner units carry a `lq^{±1/2}` half-power so this form is
    the one the N_f dressing consumes.

  * :func:`multi_arc_image` — the Weyl-normalized view matching
    `CutReglue.arc_image`: `charge -> LaurentPoly`.  Asserts integer Weyl
    exponents (raising on the half-power corner arcs, exactly as
    `CutReglue.weylize` / `skein_a1d3_kalg.weylize` do); callers that need
    the half-powers use :func:`multi_arc_ordered`.

  * :func:`select_order` — an elevation-order oracle.  The order is the
    ONLY free choice, and it is provably BENIGN (certified break scan,
    1480/1480 multi-arc charges with `|·| <= 2`: support is
    order-invariant and two orders differ only by an overall `lq^k`
    framing power).  `select_order` picks the `order_by_tri` whose ordered
    image equals a caller-supplied reference exactly — typically the
    certified global-σ product `skein_a1d3_kalg.skein_mul` of the curve's
    single-arc factors.  When the caller already knows the elevation order
    (the N_f=1 leg passes `{2: (1, 0)}` for `s_A`) it is passed directly to
    :func:`multi_arc_ordered` / :func:`multi_arc_image` and this oracle is
    unnecessary.

Self-contained: `tri` is any object exposing `triangle_edges`,
`internal_edge_ids`, `boundary_edge_ids`, `n_edges`, `n_triangles`
(and `sigma()`, for the ordered frame it is built in).  Depends only on
the triangle/Y-Delta/phi machinery — no `KAlgebra` /
`BPSKAlgebra` import, so it composes under any chart.
"""

from __future__ import annotations

from itertools import permutations, product as iproduct
from typing import Mapping, Optional, Sequence

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from laurent_poly import LaurentPoly

from curve_realization import _ARC_TYPE_BY_POSITIONS, _CCW_LATER_FIRST
from half_laurent import HalfLaurent, ONE as HL_ONE
from phi_map import phi
from triangle_algebra import TriangleAlgebra
from y_delta import project_triangle_monomial, _weyl_correction

__all__ = [
    "multi_local_realize",
    "multi_arc_triangles",
    "multi_arc_ordered",
    "multi_arc_image",
    "weylize",
    "select_order",
]


# ---------------------------------------------------------------------------
# The primitive: per-triangle arc list (no single-arc honest-fail)
# ---------------------------------------------------------------------------
def multi_local_realize(tri, charge: Sequence[int]) -> list[list[tuple]]:
    """Per-triangle list of arc descriptors of the tropical charge.

    Each entry is a list (possibly empty, length ``sum(m)``) of
    ``(arc_type, edge_at_eps, edge_at_eps_prime)`` triples, with ``m_k``
    copies of the corner-type-``k`` arc (the arc opposite ccw position
    ``k``, connecting positions ``k+1`` and ``k+2``).

    This is `curve_realization.realize_curve` WITHOUT the
    ``sum(m) > 1`` honest-fail: multi-arc triangles are returned in full.
    Parity and triangle inequality are still enforced per triangle
    (a genuinely non-realizable charge raises ``ValueError``).
    """
    out: list[list[tuple]] = []
    for ti, ecc in enumerate(tri.triangle_edges):
        a = [abs(charge[e]) for e in ecc]
        if sum(a) % 2:
            raise ValueError(f"triangle {ti}: odd parity {a}")
        m = [0, 0, 0]
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            if a[k] > a[i] + a[j]:
                raise ValueError(f"triangle {ti}: triangle ineq fails {a}")
            m[k] = (a[i] + a[j] - a[k]) // 2
        arcs: list[tuple] = []
        for k in range(3):
            for _ in range(m[k]):
                i, j = (k + 1) % 3, (k + 2) % 3
                at = _ARC_TYPE_BY_POSITIONS[frozenset({i, j})]
                lp, ep = _CCW_LATER_FIRST[at]
                arcs.append((at, ecc[lp], ecc[ep]))
        out.append(arcs)
    return out


def multi_arc_triangles(tri, charge: Sequence[int]) -> dict[int, int]:
    """`{ti: n_arcs}` for the triangles this charge crosses at least
    twice — the triangles whose elevation order is a free choice."""
    arcs_by_tri = multi_local_realize(tri, charge)
    return {ti: len(al) for ti, al in enumerate(arcs_by_tri) if len(al) >= 2}


# ---------------------------------------------------------------------------
# The quantum trace (ordered form — the certified core)
# ---------------------------------------------------------------------------
def multi_arc_ordered(
    tri,
    charge: Sequence[int],
    eps_by_bnd: Mapping[int, int],
    order_by_tri: Optional[Mapping[int, Sequence[int]]] = None,
) -> dict[tuple, HalfLaurent]:
    """Ordered-form quantum trace of a (possibly multi-arc-per-triangle)
    stated curve.

    Args:
        tri          : the ambient (bordered) triangulation.
        charge       : edge-charge vector (length ``tri.n_edges``).
        eps_by_bnd   : ``boundary edge id -> ±1``; ONE state applied to
                       every endpoint on that boundary edge (the
                       uniform-state sector — the convention the corner
                       units were certified in).  Every crossed boundary
                       edge must have a state.
        order_by_tri : ``ti -> permutation`` (tuple of indices into that
                       triangle's arc list) giving the TriangleAlgebra
                       multiplication order (= reversed elevation, since
                       `phi` is an anti-hom).  Default = identity order
                       (the arc-list order of :func:`multi_local_realize`).
                       Only multi-arc triangles are order-sensitive.

    Returns:
        ``charge -> HalfLaurent`` in the global ordered frame of
        `project_triangle_monomial` (the frame `skein_mul` lives in).
        Internal crossed edges are state-summed; the empty dict is the
        curve that dies in every sector.
    """
    arcs_by_tri = multi_local_realize(tri, charge)
    internal_states = sorted(e for e in tri.internal_edge_ids if charge[e])
    for e in tri.boundary_edge_ids:
        if charge[e] and e not in eps_by_bnd:
            raise ValueError(f"boundary edge {e} crossed but no state given")
    order_by_tri = dict(order_by_tri or {})

    out: dict[tuple, HalfLaurent] = {}
    for bits in iproduct((1, -1), repeat=len(internal_states)):
        mu = dict(zip(internal_states, bits))

        def state_at(e):
            return eps_by_bnd[e] if e in eps_by_bnd else mu[e]

        triples, coef, dead = [], HL_ONE, False
        for ti, arclist in enumerate(arcs_by_tri):
            if not arclist:
                triples.append((0, 0, 0))
                continue
            order = order_by_tri.get(ti, tuple(range(len(arclist))))
            prod = TriangleAlgebra.one()
            for idx in order:
                at, e_eps, e_epsp = arclist[idx]
                f = phi(at, state_at(e_eps), state_at(e_epsp))
                if f.is_zero():
                    dead = True
                    break
                prod = prod * f          # TriangleAlgebra ordered product
            if dead:
                break
            if prod.is_zero():
                dead = True
                break
            terms = prod.items()
            if len(terms) != 1:
                raise AssertionError(
                    f"triangle {ti}: product of phi-monomials is not a "
                    f"monomial: {terms}")
            (e_triple, co), = terms
            triples.append(e_triple)
            coef = coef * co
        if dead:
            continue
        pr = project_triangle_monomial(tri, triples, coef)
        if pr is None:
            continue
        ch, oc = pr
        s = out.get(ch, HalfLaurent.zero()) + oc
        if s.is_zero():
            out.pop(ch, None)
        else:
            out[ch] = s
    return out


def weylize(tri, charge: Sequence[int], oc: HalfLaurent) -> LaurentPoly:
    """Weyl-symmetric normal form of an ordered coefficient (verbatim
    `CutReglue.weylize` / `skein_a1d3_kalg.weylize`): shift by
    ``-_weyl_correction`` and return an integer-exponent
    :class:`LaurentPoly`.  Asserts the Weyl exponent is an integer —
    the half-power corner arcs must be read in ordered form via
    :func:`multi_arc_ordered`."""
    w = oc.shift(-_weyl_correction(tri, charge))
    d: dict[int, int] = {}
    for e, co in w.items():
        assert e.denominator == 1, f"non-integer Weyl exponent at {charge}"
        d[int(e)] = d.get(int(e), 0) + co
    return LaurentPoly(d)


def multi_arc_image(
    tri,
    charge: Sequence[int],
    eps_by_bnd: Mapping[int, int],
    order_by_tri: Optional[Mapping[int, Sequence[int]]] = None,
) -> dict[tuple, LaurentPoly]:
    """The Weyl-normalized view of :func:`multi_arc_ordered`, matching
    `CutReglue.arc_image`: ``charge -> LaurentPoly`` (integer exponents).

    This raises on the half-power corner arcs (via :func:`weylize`);
    those live in :func:`multi_arc_ordered`."""
    ordered = multi_arc_ordered(tri, charge, eps_by_bnd, order_by_tri)
    return {ch: weylize(tri, ch, oc) for ch, oc in ordered.items()
            if not oc.is_zero()}


# ---------------------------------------------------------------------------
# Elevation-order oracle
# ---------------------------------------------------------------------------
def _ordered_eq(x: Mapping[tuple, HalfLaurent],
                y: Mapping[tuple, HalfLaurent]) -> bool:
    for k in set(x) | set(y):
        if not (x.get(k, HalfLaurent.zero())
                - y.get(k, HalfLaurent.zero())).is_zero():
            return False
    return True


def select_order(
    tri,
    charge: Sequence[int],
    eps_by_bnd: Mapping[int, int],
    reference: Mapping[tuple, HalfLaurent],
) -> Optional[dict[int, tuple]]:
    """Pick the ``order_by_tri`` whose :func:`multi_arc_ordered` image
    equals ``reference`` EXACTLY (including the overall `lq^k` framing).

    ``reference`` is an ordered-form dict — typically the certified
    global-σ product `skein_a1d3_kalg.skein_mul` of the curve's
    single-arc factors, which fixes the correct framing that the
    (benign) elevation choice would otherwise leave free.

    Sweeps every permutation of every multi-arc triangle's arcs and
    returns the first matching ``order_by_tri`` (dict ``ti -> perm``,
    restricted to multi-arc triangles), or ``None`` if none matches.
    Since the order is provably benign (support-invariant, framing
    differs only by an overall `lq^k`), a match exists whenever
    ``reference`` is a valid framing of this curve."""
    multi = multi_arc_triangles(tri, charge)
    if not multi:
        img = multi_arc_ordered(tri, charge, eps_by_bnd)
        return {} if _ordered_eq(img, reference) else None
    tis = sorted(multi)
    perm_choices = [list(permutations(range(multi[ti]))) for ti in tis]
    for combo in iproduct(*perm_choices):
        order_by_tri = {ti: perm for ti, perm in zip(tis, combo)}
        img = multi_arc_ordered(tri, charge, eps_by_bnd, order_by_tri)
        if _ordered_eq(img, reference):
            return order_by_tri
    return None
