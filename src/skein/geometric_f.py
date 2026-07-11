"""Geometric quantum traces for all simple multicurves, by constructive
transport along intrinsic skein products.

Goal: the RG / chart image `Tr^lq(multicurve)` in `Y_Delta` for an
ARBITRARY simple multicurve -- the "geometric F" of the FST chart --
without the multi-arc elevation calculus (see `multicurve_trace`, which
documents why the naive multi-arc state sum is wrong).

Method (product-and-peel; the same constructive discipline as the
repo's canonical-basis constructive-build rule):

* Seed: single closed curves whose normal position crosses every
  triangle at most once -- including the peripheral loops `f_p`, which
  are always single-arc.  Their traces are computed by the verified
  single-arc state sum (`quantum_trace`).  (With the corrected
  projection these coincide with the global-Weyl characters
  `[mu_p] + [mu_p^{-1}]` on the tetrahedron; under the pre-fix
  projection they differed on two punctures -- an early symptom of
  the projection bug fixed in `y_delta.project_triangle_monomial`.)

* The quantum trace is an algebra homomorphism, and disjoint unions map
  to products, so a multicurve's trace is the product of its
  components' traces (with multiplicity) times peripheral characters.

* For an unknown single curve `c`: find a pair of *known* curves whose
  intrinsic product (`skein_algebra.SkeinAlgebra`, Kauffman resolution)
  contains `c` exactly once, bare (no peripheral dressing, multiplicity
  one) with a unit-monomial coefficient, all other terms known.  Then

      Tr(c) = coeff^{-1} * ( Tr(x) Tr(y) - sum of known terms ),

  evaluated in `Y_Delta`.  Each solved entry is overdetermined (many
  relations contain it); `verify_relation` re-checks arbitrary product
  relations and is the cross-validation between the intrinsic skein
  engine and the quantum-trace engine.

Dictionary: the Kauffman variable maps to `A -> lq` (historically only
`A^2 = lq^2` was observable — every closed curve on the SPHERE is
separating, so closed-curve products carry even A-powers only; the
sign was pinned by the `gamma_a gamma_b = A^2 gamma_c + ...`
coefficient against the `Y_Delta` product).  On higher-genus charts —
live since the 2026-07-07 id-based edge refactor, e.g. the
once-punctured torus = N=2* SU(2) — odd intersection numbers produce
odd A-powers, and the `A -> +lq` convention (already hardwired in the
peel's inverter) is pinned by the `verify_relation` cross-validation
there (`tests/test_edge_data_engine.py`).  The trivial loop is
`delta = -(lq^2 + lq^{-2})`.

Self-contained within `skein_sphere`.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations_with_replacement
from typing import Sequence

from half_laurent import HalfLaurent
from multicurve import components_of_coords, is_admissible, normal_arcs
from quantum_trace import quantum_trace
from skein_algebra import APoly, SkeinAlgebra, SkeinElement
from triangulation import Triangulation
from y_delta import YDelta


Coords = tuple[int, ...]


def apoly_to_lq(c: APoly) -> HalfLaurent:
    """Convert a Kauffman coefficient to `lq` via `A -> lq`.

    On the SPHERE every simple closed curve is separating, so
    closed-curve products only produce even A-powers and only
    `A^2 = lq^2` is observable (the historically pinned dictionary).
    On higher-genus charts (live since the id-based edge refactor,
    2026-07-07: e.g. the once-punctured torus = N=2* SU(2)) curves
    with odd intersection number produce odd A-powers, so the sign
    convention `A -> +lq` becomes visible.  It matches the convention
    already hardwired in the product-and-peel inverter
    (`GeometricF._grow_until`), and is pinned empirically by the
    engine-vs-quantum-trace cross-validation `verify_relation` on the
    torus (the two sign choices differ by the standard multicurve
    rescaling, so consistency is the meaningful check).
    """
    out: dict = {}
    for e, v in c.items():
        out[e] = out.get(e, 0) + v
    return HalfLaurent({e: v for e, v in out.items()})


class GeometricF:
    """Lazy table of quantum traces of simple multicurves on a chart."""

    def __init__(self, triangulation: Triangulation, *, skein: SkeinAlgebra | None = None):
        self.t = triangulation
        self.skein = skein or SkeinAlgebra(triangulation)
        self._single: dict[Coords, YDelta] = {}
        self._seeded = False

    # -- seeding -------------------------------------------------------

    def _seed(self) -> None:
        """Enumerate the single-arc-realizable closed curves and store
        their state-sum traces."""
        if self._seeded:
            return
        t = self.t
        n = t.n_edges
        for mask in range(1, 1 << n):
            coords = tuple((mask >> e) & 1 for e in range(n))
            if not is_admissible(t, coords):
                continue
            arcs = normal_arcs(t, coords)
            if any(len(tri) > 1 for tri in arcs):
                continue
            comps = components_of_coords(t, coords)
            if len(comps) != 1:
                continue
            self._single[coords] = quantum_trace(t, coords)
        # peripheral loops are single-arc but may have coords outside
        # the 0/1 mask range on other triangulations; seed explicitly.
        for p in range(t.n_punctures):
            fp = t.puncture_flavour_charge(p)
            if fp not in self._single:
                arcs = normal_arcs(t, fp)
                if all(len(tri) <= 1 for tri in arcs):
                    self._single[fp] = quantum_trace(t, fp)
        self._seeded = True

    # -- public evaluation ----------------------------------------------

    def trace_single(self, coords: Sequence[int], *, max_relations: int = 4000) -> YDelta:
        """Trace of a single (connected, non-peripheral) closed curve."""
        coords = tuple(coords)
        self._seed()
        if coords in self._single:
            return self._single[coords]
        self._grow_until(coords, max_relations=max_relations)
        if coords not in self._single:
            raise RuntimeError(
                f"product-and-peel search did not reach curve {coords}; "
                "raise max_relations or extend the seeding"
            )
        return self._single[coords]

    def trace_label(self, core: Sequence[int], peripherals: Sequence[int]) -> YDelta:
        """Trace of a basis multicurve label `(core, k)`:
        product of component traces times peripheral characters."""
        t = self.t
        core = tuple(core)
        out = YDelta.one(t)
        for comp, mult in Counter(components_of_coords(t, core)).items():
            tr = self.trace_single(comp)
            for _ in range(mult):
                out = out * tr
        for p, kp in enumerate(peripherals):
            if not kp:
                continue
            fp = t.puncture_flavour_charge(p)
            tr = self.trace_single(fp)
            for _ in range(int(kp)):
                out = out * tr
        return out

    def trace_element(self, x: SkeinElement) -> YDelta:
        """Trace of a Z[A^{+/-1}]-combination of basis multicurves."""
        out = YDelta.zero(self.t)
        for (core, k), c in x.terms.items():
            out = out + apoly_to_lq(c) * self.trace_label(core, k)
        return out

    # -- the peel --------------------------------------------------------

    def _known_label(self, core: Coords, k: tuple[int, ...]) -> bool:
        return all(
            comp in self._single
            for comp in set(components_of_coords(self.t, core))
        )

    def _grow_until(self, target: Coords, *, max_relations: int) -> None:
        """Breadth-first product search: multiply pairs of known curves,
        solve every relation with exactly one bare unknown, repeat."""
        t = self.t
        used = 0
        progress = True
        while target not in self._single and progress and used < max_relations:
            progress = False
            known = sorted(self._single, key=lambda c: (sum(c), c))
            for x, y in combinations_with_replacement(known, 2):
                if used >= max_relations:
                    break
                used += 1
                prod = self.skein.multiply_labels(
                    (x, (0,) * t.n_punctures), (y, (0,) * t.n_punctures),
                )
                unknown: Coords | None = None
                ok = True
                for (core, k), coeff in prod.terms.items():
                    comps = Counter(components_of_coords(t, core))
                    for comp, mult in comps.items():
                        if comp in self._single:
                            continue
                        if unknown is not None and unknown != comp:
                            ok = False
                            break
                        # solvability: bare, multiplicity one, unit coeff
                        if (mult != 1 or any(k) or len(comps) != 1
                                or len(list(coeff.items())) != 1
                                or abs(list(coeff.items())[0][1]) != 1):
                            ok = False
                            break
                        unknown = comp
                    if not ok:
                        break
                if not ok or unknown is None:
                    continue
                # solve for the unknown
                lhs = self._single[x] * self._single[y]
                rhs_known = YDelta.zero(t)
                unit_coeff: APoly | None = None
                for (core, k), coeff in prod.terms.items():
                    comps = Counter(components_of_coords(t, core))
                    if unknown in comps:
                        unit_coeff = coeff
                        continue
                    rhs_known = rhs_known + apoly_to_lq(coeff) * self.trace_label(core, k)
                assert unit_coeff is not None
                ((e, v),) = unit_coeff.items()
                inv = HalfLaurent({-e: v})    # v = +/-1
                self._single[unknown] = inv * (lhs - rhs_known)
                progress = True
                if unknown == target:
                    return

    # -- cross-validation -------------------------------------------------

    def verify_relation(self, coords_x: Sequence[int], coords_y: Sequence[int]) -> bool:
        """Check `Tr(x) Tr(y) == Tr(intrinsic product x*y)` in Y_Delta.

        Requires (and triggers) table coverage of every curve in the
        decomposition.  This is the load-bearing consistency check
        between the Kauffman resolution engine and the quantum trace.
        """
        t = self.t
        x = tuple(coords_x)
        y = tuple(coords_y)
        lhs = self.trace_label(*self.skein.label_of_coords(x)) * \
            self.trace_label(*self.skein.label_of_coords(y))
        prod = self.skein.multiply_labels(
            self.skein.label_of_coords(x), self.skein.label_of_coords(y),
        )
        rhs = self.trace_element(prod)
        return lhs == rhs
