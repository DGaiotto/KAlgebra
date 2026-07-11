"""`SkeinSphereKAlg` -- the closed n-punctured sphere Sk(S^2_{0,n}) as a
KAlgebra, on a skein-nice FST triangulation.

It is one object presenting the closed-sphere SkeinKAlgebra with BOTH
surfaces.

* **Canonical contract** -- it IS a `BPSKAlgebra` on the FST exchange
  matrix `sigma_Delta = triangulation.sigma()` (a complete RGKAlgebra),
  so `multiply` / `rho` / `trace` / `inner_product` / the verifiers and
  the **auto-found spectrum generator** (`spec` / `rg_generator`) are all
  inherited.  On the closed sphere the auto-found spec IS the "figure out
  S" answer: S^2_{0,n} is an SU(2)^{n-3} gauge theory, so its
  spectrum is the gauge-theory spectrum.

* **Intrinsic skein surface** -- the topological `SkeinAlgebra`
  (multicurve basis, Kauffman multiply) and the **geometric F**
  (`GeometricF`, the S-FREE quantum trace) are exposed: `curve`,
  `multiply_skein`, `geometric_F`.  The geometric F is the bare
  (unit-coefficient) part of the canonical `F(-gamma)`; the difference is
  the monopole-bubbling **dressing** (`dressing`), whose closed
  geometric form is the open prize.

Tetrahedron `S^2_{0,4}` = SU(2) N_f=4 builds in ~0.3s; geometric F of a
generator curve is ~0s.  (The canonical `multiply`/`trace` inherited from
BPS use the F-solve / Schur-Nahm sum and are correct-but-expensive; the
geometric F is the fast intrinsic shortcut.)  The full given-F/solve-S
`SkeinKAlgebra(RGKAlgebra)` (RG overridden to the geometric F, S from
mutations) is the deeper refinement; this object consolidates the
canonical surface + the skein bridge on the closed sphere.

Run:  PYTHONPATH=. python -c "from skein_sphere_kalg import \
SkeinSphereKAlg; A=SkeinSphereKAlg.tetrahedron(); print(A.spec)"
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bps_kalgebra import BPSKAlgebra          # repo root
from kalgebra import Element
from laurent_poly import LaurentPoly

from triangulation import Triangulation
from skein_algebra import SkeinAlgebra
from geometric_f import GeometricF


class SkeinSphereKAlg(BPSKAlgebra):
    """Closed-sphere skein algebra as a KAlgebra (canonical = BPS on
    sigma_Delta; intrinsic skein + geometric F added)."""

    def __init__(self, triangulation: Triangulation, *, spec=None):
        # The skein engine (multicurves / Kauffman / geometric F) accepts
        # from_edge_data charts (self-loops + multi-edges) too.  The
        # canonical contract additionally needs a spectrum generator: pass
        # `spec` (e.g. transported from the SU(2)-gauging recursion) or
        # rely on auto-find where feasible.
        self.triangulation = triangulation
        self.skein = SkeinAlgebra(triangulation)
        self._geom = GeometricF(triangulation)
        n = triangulation.n_edges
        sigma = triangulation.sigma()
        nodes = [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]
        # Canonical contract + spec.  `spec=None` => BPSKAlgebra auto-finds
        # (prohibitive for n>=6); pass a sigma_Delta-FRAME spec to skip it
        # (cached, or transported from the SU(2)-gauging recursion -- the
        # systematic n>5 route).
        super().__init__(pairing=sigma, node_charges=nodes, spec=spec)

    # -- factories -----------------------------------------------------
    @classmethod
    def tetrahedron(cls) -> "SkeinSphereKAlg":
        """S^2_{0,4} = SU(2) N_f=4 (6 edges, gauge rank 1, Spin(8) flavour)."""
        return cls(Triangulation.tetrahedron_S2_4())

    @classmethod
    def bipyramid(cls) -> "SkeinSphereKAlg":
        """S^2_{0,5} bipyramid (9 edges, SU(2)^2 sausage; slow auto-find)."""
        return cls(Triangulation.bipyramid_S2_5())

    # -- SU(2)^n flavour (one SU(2) per puncture) ----------------------
    def su2n_flavoured(self, *, strict: bool = True):
        """This closed-sphere algebra re-presented over **SU(2)^n** (one
        SU(2) per puncture -- the manifest non-abelian flavour), instead of
        the abelian `U(1)^n` Cartan the BPS realisation exposes.  Returns a
        `SU2nFlavouredKAlgebra`: structure (`multiply`/`rho`) is unchanged,
        the flavour content of `trace`/`inner_product` is un-branched into
        SU(2)^n multiplets (`TensorZPlusRing([SU2ZPlusRing()] * n)`)."""
        from su2n_flavour import SU2nFlavouredKAlgebra
        return SU2nFlavouredKAlgebra(self, self.coefficient_ring().rank,
                                     strict=strict)

    # -- intrinsic skein surface (fast, S-free) ------------------------
    def curve(self, coords):
        """The skein basis multicurve with the given admissible coords."""
        return self.skein.curve(coords)

    def multiply_skein(self, a_coords, b_coords):
        """Intrinsic Kauffman product of two basis multicurves (BARE --
        not the canonical/dressed product; that is inherited `multiply`)."""
        npunct = self.triangulation.n_punctures
        return self.skein.multiply_labels(
            (tuple(a_coords), (0,) * npunct),
            (tuple(b_coords), (0,) * npunct),
        )

    def geometric_F(self, curve_coords) -> Element:
        """The S-FREE geometric F (quantum trace) of a connected curve, as
        a quantum-torus `Element` in the auxiliary convention (weyl
        coefficient lq^e |-> q^e, charges [y^a] |-> X_a).  Fast (~0s for a
        generator); = the bare part of the canonical F(-gamma)."""
        qt = self._geom.trace_single(tuple(curve_coords))
        terms = {}
        for c in qt._terms:
            ((e, co),) = qt.weyl_coefficient(c).items()
            terms[tuple(c)] = LaurentPoly({int(e): co})
        return Element(terms)

    def dressing(self, curve_coords) -> Element:
        """The monopole-bubbling dressing of a curve = canonical F(-gamma)
        minus the geometric (bare) F.  Uses the canonical F-solve
        (expensive); the closed geometric form of this is the open prize."""
        g = tuple(curve_coords)
        ng = tuple(-x for x in g)
        full = self.F(ng)                       # canonical F (slow F-solve)
        bare = self.geometric_F(g)
        terms = dict(full)
        for c, co in bare.terms.items():
            terms[c] = terms.get(c, LaurentPoly.zero()) - co
        return Element({c: v for c, v in terms.items() if v != LaurentPoly.zero()})
