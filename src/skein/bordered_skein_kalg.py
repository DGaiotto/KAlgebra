"""`BorderedSkeinKAlg` -- the SkeinSphereKAlg approach extended to
IRREGULAR PUNCTURES by *not fully gluing all edges*.

Sibling of `SkeinSphereKAlg` (the closed n-punctured sphere): where
`SkeinSphereKAlg` requires a CLOSED FST triangulation (every edge glued),
`BorderedSkeinKAlg` takes a `BorderedTriangulation` with SOME edges left
unglued.  An unglued edge is a **boundary marked interval** -- the data of
an irregular puncture (rank ~ #marks).  So a bordered surface = a sphere carrying
both regular punctures (glued vertices) and irregular punctures (boundary
circles with marks).

Construction (mutable block + sidecar):

* **Canonical algebra = `BPSKAlgebra` on the MUTABLE internal block** of
  the FST exchange matrix `sigma_Delta~` (the interior edges = the BPS
  quiver; the unglued boundary edges are frozen).  The full canonical
  contract -- `multiply` / `rho` / `trace` / `inner_product` / the
  verifiers / the auto-found spectrum generator -- is inherited.  This is
  the literal SkeinSphereKAlg recipe (`BPSKAlgebra(sigma)`), now on the
  bordered exchange matrix.

* **Boundary `Z^d` sidecar** -- the unglued boundary edges and how they
  pair into the mutable quiver (`boundary_pairing()`) are carried as
  metadata.  They are NOT needed by the canonical algebra (an even-marked
  irregular puncture's U(1) flavour mu already lives in the mutable
  block's kernel); their role is to host the **stated skein surface**
  (open arcs ending on boundary edges), the S-free geometric-F oracle.
  For fan-POLYGON (disk) charts this is wired (`stated_geometric_F` =
  `kappa_Delta` into the bordered Y_Delta~; the pentagon Muller diagonals
  reproduce the chart F); for the
  general cut-sphere charts the bordered stated engine is a follow-up.

Why this is robust (vs the "unpin" route
`SkeinPentagonKAlg`/`SkeinU1HexagonKAlg`): cutting an FST edge **deletes
that node from the quiver**, so the canonical algebra is exactly the
node-drop RG IR theory -- the Le stated-skein edge-cut.  The charts stay
unimodular (FST unit edge
vectors), avoiding the non-unimodular `[[0,2],[-2,0]]` presentation that
needs an ambient lift.

Examples:

    BorderedSkeinKAlg.polygon(5)        # disk, 5 marks = A_2 AD (1 irreg)
    BorderedSkeinKAlg.cut_tetrahedron() # S^2: 3 reg + 1 rank-1/2 irreg
                                        #   = SU(2) N_f=3 (flavour SO(6))

Run:  PYTHONPATH=. python -c "from bordered_skein_kalg import \
BorderedSkeinKAlg; A=BorderedSkeinKAlg.polygon(5); print(A.multiply((1,0),(0,1)))"
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bps_kalgebra import BPSKAlgebra          # repo root

from bordered_triangulation import BorderedTriangulation
from triangulation import Triangulation


class BorderedSkeinKAlg(BPSKAlgebra):
    """Bordered (irregular-puncture) skein algebra as a KAlgebra:
    canonical = BPSKAlgebra on the mutable internal block of the bordered
    FST exchange matrix; boundary edges carried as the `Z^d` sidecar."""

    def __init__(self, bordered: BorderedTriangulation, *, spec=None, verify="on"):
        self.bordered = bordered
        sigma_full = bordered.sigma()
        slot_to_edge, internal_ids, boundary_ids = bordered.edge_numbering()
        self._sigma_full = sigma_full
        self._internal_ids = internal_ids
        self._boundary_ids = boundary_ids
        self._slot_to_edge = slot_to_edge

        if not internal_ids:
            raise NotImplementedError(
                "BorderedSkeinKAlg needs at least one interior (glued) edge: "
                "a fully-unglued polygon (e.g. the single triangle, "
                "fan_polygon(3)) has an empty BPS quiver -- that is the pure "
                "stated-skein object (stated_triangle), not a BPSKAlgebra."
            )

        mb = [[sigma_full[i][j] for j in internal_ids] for i in internal_ids]
        nodes = [tuple(1 if k == i else 0 for k in range(len(internal_ids)))
                 for i in range(len(internal_ids))]
        # spec=None => BPSKAlgebra auto-finds (fine for the small certified
        # charts; pass a mutable-frame spec to skip auto-find for big ones).
        super().__init__(pairing=mb, node_charges=nodes, spec=spec, verify=verify)

        # Detect the fan-polygon (disk) case: the intrinsic STATED skein
        # surface (`stated_polygon` engine) is defined there.  Cut-sphere
        # charts need the general bordered stated engine (deferred).
        nb = len(bordered.boundary_sides)
        self._stated_n = (
            nb if nb >= 4
            and bordered == BorderedTriangulation.fan_polygon(nb)
            else None
        )

    # -- factories ------------------------------------------------------
    @classmethod
    def polygon(cls, n: int, *, spec=None) -> "BorderedSkeinKAlg":
        """The fan-triangulated n-gon = the disk with n marked points = the
        A_{n-3} Argyres-Douglas theory (one irregular puncture).  Needs
        n >= 4 (n=3 is the single triangle, no quiver)."""
        if n < 4:
            raise ValueError(
                f"polygon(n) needs n >= 4 for a nonempty BPS quiver; "
                f"n=3 is the bare stated triangle.  Got n={n}."
            )
        return cls(BorderedTriangulation.fan_polygon(n), spec=spec)

    @classmethod
    def from_cut_closed(
        cls, tri: Triangulation, cut_edge_ids, *, spec=None
    ) -> "BorderedSkeinKAlg":
        """Build from a CLOSED Triangle-based triangulation with the given
        FST edges *cut* (un-glued) -- the "not fully gluing all edges"
        operation that creates irregular punctures."""
        return cls(BorderedTriangulation.from_closed(tri, cut_edge_ids),
                   spec=spec)

    @classmethod
    def cut_tetrahedron(cls, cut_edge_ids=(4,), *, spec=None) -> "BorderedSkeinKAlg":
        """The tetrahedral S^2_{0,4} = SU(2) N_f=4 with the given FST edges
        cut.  The SU(2) N_f tower (gauge rank 1, flavour 4 - #cuts):
        ``()`` N_f=4 (SO(8)), ``(4,)`` N_f=3 (SO(6)), ``(4,5)`` N_f=2
        (SO(4)), ``(4,5,0)`` N_f=1 (SO(2)).  Default cuts edge 4 = (1-3):
        SU(2) N_f=3."""
        return cls.from_cut_closed(
            Triangulation.tetrahedron_S2_4(), cut_edge_ids, spec=spec)

    @classmethod
    def cut_bipyramid(cls, cut_edge_ids=(0,), *, spec=None) -> "BorderedSkeinKAlg":
        """The bipyramid S^2_{0,5} = SU(2)^2 sausage with the given FST
        edges cut: a **multi-node** SU(2)^2 gauge theory with irregular
        punctures (gauge rank 2; flavour 5 - #cuts).  Cutting collapses the
        chamber auto-find -- the closed bipyramid is ~200s, a 1-cut ~3s."""
        return cls.from_cut_closed(
            Triangulation.bipyramid_S2_5(), cut_edge_ids, spec=spec)

    @classmethod
    def cut_octahedron(cls, cut_edge_ids=(0,), *, spec=None) -> "BorderedSkeinKAlg":
        """The octahedron S^2_{0,6} = SU(2)^3 sausage with the given FST
        edges cut: a **3-node** SU(2)^3 gauge theory with irregular
        punctures (gauge rank 3; flavour 6 - #cuts) -- the next multi-node
        scaling case after `cut_bipyramid` (SU(2)^2).  The build leans on
        the auto-find collapse (cuts are far cheaper than the closed chart);
        for big charts the spec can be supplied."""
        return cls.from_cut_closed(
            Triangulation.octahedron_S2_6(), cut_edge_ids, spec=spec)

    # -- boundary (Z^d) sidecar -----------------------------------------
    @property
    def n_interior_edges(self) -> int:
        return len(self._internal_ids)

    @property
    def n_boundary_edges(self) -> int:
        return len(self._boundary_ids)

    def full_sigma(self) -> list[list[int]]:
        """The FST exchange matrix over ALL edges (interior + boundary) --
        the Gamma~ = Gamma_int (+) Z^d lattice pairing.  Interior edges are
        indices ``range(n_interior_edges)``; boundary edges follow."""
        return [row[:] for row in self._sigma_full]

    def boundary_pairing(self) -> list[list[int]]:
        """The interior x boundary block of `full_sigma()`: how each frozen
        boundary direction pairs into the mutable quiver (the Gamma~
        coupling the stated arcs will use)."""
        bi = self._boundary_ids
        return [[self._sigma_full[i][j] for j in bi] for i in self._internal_ids]

    # -- intrinsic stated skein surface --------------------------------
    @property
    def is_fan_polygon(self) -> bool:
        """Whether this is a fan-polygon (disk) chart -- the case where the
        intrinsic stated skein surface (`stated_polygon` engine) is defined."""
        return self._stated_n is not None

    def stated_polygon(self):
        """The `StatedPolygon` carrying the intrinsic stated skein algebra
        (open arcs ending on the boundary edges with +/- states)."""
        if self._stated_n is None:
            raise NotImplementedError(
                "the intrinsic stated skein surface is wired for the "
                "fan-POLYGON (disk) charts (the case the stated_polygon engine "
                "covers); for cut-sphere charts it needs the general bordered "
                "stated engine, which is not yet implemented."
            )
        from stated_polygon import StatedPolygon
        return StatedPolygon(self._stated_n)

    def stated_geometric_F(self, s: int, t: int, eps_s: int = 1, eps_t: int = 1):
        """The **S-free** stated quantum trace (geometric F) of the stated
        boundary arc from marked point `s` to `t` with end states
        `eps_s, eps_t in {+1,-1}` -- `kappa_Delta = (otimes phi) o rho`
        landing in `(x)_T Y(T)` (= the bordered Chekhov-Fock torus
        `Y_Delta~` after the internal-edge gluing).  This is the intrinsic
        skein surface for the bordered/irregular case: an algebra
        homomorphism, computed with NO spectrum generator (cf. the closed
        `SkeinSphereKAlg.geometric_F`).  Fan-polygon charts only.

        The interior-edge exponents (`stated_internal_exps`) are the image
        in the mutable-block charges; for the pentagon the Muller-diagonal
        images reproduce the chart canonical F exactly (the stated <-> chart
        bridge)."""
        from kappa_qt import kappa
        P = self.stated_polygon()
        return kappa(P, P.arc(s % self._stated_n, t % self._stated_n, eps_s, eps_t))

    def stated_internal_exps(self, key):
        """Read the per-internal-edge exponent of a `stated_geometric_F`
        term, asserting the two glued copies agree (Chekhov-Fock
        well-formedness of the image in `Y_Delta~`)."""
        from kappa_qt import internal_exps
        return internal_exps(self.stated_polygon(), key)

    # the closed-curve multicurve `GeometricF` does not apply on a bordered
    # surface; the stated surface above replaces it for polygon charts.
    def geometric_F(self, *args, **kwargs):
        raise NotImplementedError(
            "use `stated_geometric_F` (the stated open-arc quantum trace) on "
            "bordered charts; the closed multicurve `GeometricF` is for the "
            "closed-sphere `SkeinSphereKAlg`."
        )
