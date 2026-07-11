"""`SkeinSphereAtlas` — the triangulation-labelled `BPSAtlas`:
charts = ideal triangulations of the closed punctured surface,
transitions = diagonal FLIPS (= quiver mutations at the flipped edge).

Integration of the id-based edge representation with the canonical BPS
machinery:

  * the root is a `SkeinSphereKAlg` (a complete `BPSKAlgebra` on the
    FST exchange matrix sigma_Delta, carrying the intrinsic skein
    surface: multicurves, Kauffman multiply, the S-free geometric F);
  * `flip(edge)` mutates the underlying `BPSAtlas` at node `edge` AND
    flips the triangulation, CERTIFYING per flip that the mutated
    chart's node-basis exchange matrix equals the flipped
    triangulation's sigma (the Fomin-Shapiro-Thurston flip/mutation
    compatibility, checked exactly on every transition — never
    assumed);
  * every chart therefore carries BOTH surfaces natively:
    `chart_kalg(key)` rebuilds the chart as a first-class
    `SkeinSphereKAlg` on the flipped triangulation with the
    FLIP-TRANSPORTED spec (necklaced along the mutation path and
    recoordinatized to the chart's node basis — no spec auto-find on
    any non-root chart), and `chart_iso(key)` returns the certified
    `KAlgebraIso` root -> native chart (the atlas transition composed
    with the unimodular node-basis recoordinatization, the
    `bps_frame_change` pattern);
  * `verify_bare_part(coords, key)` checks, per chart, that the
    S-free geometric F equals the unit/bare part of the canonical
    BPS `F(-gamma)` — the quantum-trace <-> F dictionary, now
    certified on flipped (multi-edge) charts too.

Flips that would create self-folded triangles honest-fail (engine
scope); flips whose spec does not cooperate within the local-move
budget raise from the chart graph.

Run a quick demo:
    PYTHONPATH=. python -c "
    from skein_sphere_atlas import SkeinSphereAtlas
    at = SkeinSphereAtlas.tetrahedron()
    key, iso = at.flip(0)
    print(at.verify_bare_part((0,1,1,1,1,0)))       # root chart
    "
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bps_atlas import BPSAtlas
from bps_kalgebra import BPSKAlgebra
from directional_subquiver_rg import _node_basis_solver
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly

from triangulation import Triangulation
from skein_sphere_kalg import SkeinSphereKAlg

_ONE = LaurentPoly({0: 1})


class SkeinSphereAtlas:
    """Ensemble of triangulation charts of one `Sk(S)`-K-algebra, with
    certified flip transitions (see module docstring)."""

    def __init__(self, triangulation: Triangulation, *, spec=None,
                 max_local_moves: int = 8, **atlas_kwargs):
        self.root_kalg = SkeinSphereKAlg(triangulation, spec=spec)
        self.atlas = BPSAtlas(self.root_kalg,
                              max_local_moves=max_local_moves,
                              **atlas_kwargs)
        self._tri: dict[tuple, Triangulation] = {(): triangulation}
        self._native: dict[tuple, SkeinSphereKAlg] = {(): self.root_kalg}

    # -- factories -------------------------------------------------------
    @classmethod
    def tetrahedron(cls, **kw) -> "SkeinSphereAtlas":
        return cls(Triangulation.tetrahedron_S2_4(), **kw)

    @classmethod
    def bipyramid(cls, *, spec=None, **kw) -> "SkeinSphereAtlas":
        return cls(Triangulation.bipyramid_S2_5(), spec=spec, **kw)

    # -- accessors ---------------------------------------------------------
    def keys(self):
        return list(self._tri)

    def triangulation(self, key=()) -> Triangulation:
        return self._tri[BPSAtlas._norm_key(key)]

    def chart(self, key=()):
        """The atlas chart (root-frame `BPSKAlgebra`) at `key`."""
        return self.atlas.chart(key)

    def iso(self, src=(), dst=()) -> KAlgebraIso:
        """Certified transition `src -> dst` in root-frame labels."""
        return self.atlas.iso(src, dst)

    # -- the flip transition ----------------------------------------------
    def _node_exchange(self, key) -> list[list[int]]:
        A = self.atlas.chart(key)
        nodes = [tuple(g) for g in A.node_charges]
        br = A.lattice.bracket
        n = len(nodes)
        return [[br(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]

    def flip(self, edge: int, key=()) -> "tuple[tuple, KAlgebraIso]":
        """Flip the triangulation at `edge` (= mutate the chart at node
        `edge`); returns `(new_key, iso)` with the composed certified
        `root -> new_key` iso.  Certifies flip == mutation exactly."""
        key = BPSAtlas._norm_key(key)
        tri2 = self._tri[key].flip(edge)      # honest-fails on self-folded
        last = None
        for d in ("fwd", "inv"):
            try:
                new_key, iso = self.atlas.mutate(key, edge, d)
                break
            except ValueError as exc:
                last = exc
        else:
            raise last
        got = self._node_exchange(new_key)
        want = tri2.sigma()
        if got != want:
            raise RuntimeError(
                f"flip/mutation mismatch at edge {edge} of chart {key}: "
                f"node-basis exchange != flipped sigma"
            )
        self._tri[new_key] = tri2
        return new_key, iso

    def flip_path(self, edges, key=()) -> tuple:
        """Apply a sequence of flips; returns the end chart key."""
        key = BPSAtlas._norm_key(key)
        for e in edges:
            key, _ = self.flip(e, key)
        return key

    # -- the native chart algebra (both surfaces on the flipped chart) ----
    def _solver(self, key):
        A = self.atlas.chart(key)
        return _node_basis_solver([tuple(g) for g in A.node_charges])

    def chart_spec(self, key) -> list[tuple]:
        """The chart's spectrum generator in ITS OWN node basis — the
        root spec necklaced along the mutation path, recoordinatized.
        This is the flip-transport that replaces spec auto-find."""
        A = self.atlas.chart(key)
        solver = self._solver(key)
        return [tuple(solver(tuple(g))) for g in A.spec]

    def chart_kalg(self, key=()) -> SkeinSphereKAlg:
        """The chart as a first-class `SkeinSphereKAlg` on the flipped
        triangulation (canonical contract + intrinsic skein surface +
        S-free geometric F), spec supplied by flip transport."""
        key = BPSAtlas._norm_key(key)
        if key not in self._native:
            self._native[key] = SkeinSphereKAlg(
                self._tri[key], spec=self.chart_spec(key))
        return self._native[key]

    def chart_iso(self, key=()) -> KAlgebraIso:
        """Certified `KAlgebraIso` root_kalg -> chart_kalg(key): the
        atlas transition composed with the unimodular node-basis
        recoordinatization (the `bps_frame_change` pattern — same
        chart, parallel coordinates)."""
        key = BPSAtlas._norm_key(key)
        native = self.chart_kalg(key)
        if not key:
            return KAlgebraIso.identity(native, name="skein-atlas[root]")
        A = self.atlas.chart(key)
        atlas_iso = self.atlas.iso((), key)
        solver = self._solver(key)
        nodes = [tuple(int(x) for x in g) for g in A.node_charges]
        rank = len(nodes[0])

        def to_nb(root_label):
            return tuple(solver(tuple(root_label)))

        def from_nb(nb_label):
            return tuple(
                sum(int(c) * nodes[i][j] for i, c in enumerate(nb_label))
                for j in range(rank)
            )

        def fwd(root_label):
            img = atlas_iso.map(Element({tuple(root_label): _ONE}))
            return Element({to_nb(l): c for l, c in img.terms.items()})

        def inv(nb_label):
            return atlas_iso.inverse(Element({from_nb(nb_label): _ONE}))

        return KAlgebraIso(self.root_kalg, native, fwd, inv,
                           name=f"skein-atlas[{BPSAtlas._key_str(key)}]")

    # -- per-chart certification: geometric F = bare part of BPS F --------
    def verify_bare_part(self, curve_coords, key=()) -> bool:
        """On chart `key`'s native algebra: the S-free geometric F of a
        CORE (non-peripheral) curve equals the restriction of the
        canonical `F(-gamma)` to the skein support (the quantum-trace =
        bare-F dictionary), with every skein term matched exactly.

        Scope: connected non-peripheral curves.  Peripheral loops are
        flavour content — their skein trace is the 2-term central
        character `[f] + [-f]` while the Z-form canonical at a flavour
        charge is the single monomial `X_{-f}` (the flavour-in-labels
        convention), so the bare-part dictionary does not apply there;
        peripheral input raises `ValueError`."""
        from multicurve import peripheral_split

        native = self.chart_kalg(key)
        g = tuple(curve_coords)
        core, k = peripheral_split(native.triangulation, g)
        if not any(core):
            raise ValueError(
                f"curve {g} is peripheral (k={k}); the bare-part "
                "dictionary applies to core curves only"
            )
        bare = native.geometric_F(g).terms
        full = native.F(tuple(-x for x in g))
        return {c: v for c, v in full.items() if c in bare} == bare


class BorderedSkeinAtlas:
    """The triangulation-labelled `BPSAtlas` for BORDERED charts
    (irregular singularities): charts = id-based bordered
    triangulations (`Triangulation.from_edge_data(allow_boundary=True)`,
    e.g. `Triangulation.cut(e)` of a closed chart or
    `Triangulation.from_bordered_slots(fan_polygon(n))`), transitions =
    diagonal flips at INTERNAL edges (boundary edges are frozen — a
    flip there raises).

    The canonical algebra per chart is `BPSKAlgebra` on the MUTABLE
    internal block of the bordered FST exchange matrix (the
    `BorderedSkeinKAlg` recipe; the boundary block rides along as the
    `Z^d` sidecar and the even-marks U(1) mu lives in the mutable
    block's kernel).  Each flip is certified TWICE:

      * `flip_e(Delta).sigma() == mu_e(Delta.sigma())` over the FULL
        (frozen-extended) matrix — the bordered FST flip/mutation
        compatibility, boundary rows transforming by the same rule;
      * the mutated BPS chart's node-basis exchange equals the flipped
        chart's mutable block.

    Spec transports along flips (necklace), so no chart off the root
    pays an auto-find.  Since edge ids are stable under flips, the
    internal-edge id set is chart-independent and node index i always
    means internal edge `internal_ids[i]`.
    """

    def __init__(self, triangulation: Triangulation, *, spec=None,
                 max_local_moves: int = 8, verify: str = "on",
                 **atlas_kwargs):
        if not triangulation.is_bordered:
            raise ValueError(
                "BorderedSkeinAtlas needs a bordered chart; use "
                "SkeinSphereAtlas for closed surfaces"
            )
        self._internal_ids = list(triangulation.internal_edge_ids)
        if not self._internal_ids:
            raise NotImplementedError(
                "no internal edges: the fully-unglued chart is the pure "
                "stated object, not a BPSKAlgebra"
            )
        mb = triangulation.mutable_block()
        n = len(mb)
        self.root_kalg = BPSKAlgebra(
            pairing=mb,
            node_charges=[tuple(1 if j == i else 0 for j in range(n))
                          for i in range(n)],
            spec=spec, verify=verify,
        )
        self.atlas = BPSAtlas(self.root_kalg,
                              max_local_moves=max_local_moves,
                              **atlas_kwargs)
        self._tri: dict[tuple, Triangulation] = {(): triangulation}
        self._native: dict[tuple, BPSKAlgebra] = {(): self.root_kalg}

    # -- accessors ---------------------------------------------------------
    def keys(self):
        return list(self._tri)

    def triangulation(self, key=()) -> Triangulation:
        return self._tri[BPSAtlas._norm_key(key)]

    def chart(self, key=()):
        return self.atlas.chart(key)

    def iso(self, src=(), dst=()) -> KAlgebraIso:
        return self.atlas.iso(src, dst)

    def node_of_edge(self, edge: int) -> int:
        """The mutable node index of internal edge id `edge`."""
        return self._internal_ids.index(edge)

    # -- the flip transition ----------------------------------------------
    def _node_exchange(self, key) -> list[list[int]]:
        A = self.atlas.chart(key)
        nodes = [tuple(g) for g in A.node_charges]
        br = A.lattice.bracket
        n = len(nodes)
        return [[br(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]

    @staticmethod
    def _mutate_matrix(B, k):
        n = len(B)
        out = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == k or j == k:
                    out[i][j] = -B[i][j]
                else:
                    s = 1 if B[i][k] > 0 else (-1 if B[i][k] < 0 else 0)
                    out[i][j] = B[i][j] + (
                        s * max(0, B[i][k] * B[k][j]) if s else 0)
        return out

    def flip(self, edge: int, key=()) -> "tuple[tuple, KAlgebraIso]":
        """Flip internal `edge` (= mutate the mutable node at that edge);
        certifies the bordered FST identity on the FULL frozen-extended
        matrix AND the mutable block per transition."""
        key = BPSAtlas._norm_key(key)
        tri = self._tri[key]
        tri2 = tri.flip(edge)     # ValueError on boundary; NIE on self-folded
        if tri2.sigma() != self._mutate_matrix(tri.sigma(), edge):
            raise RuntimeError(
                f"bordered flip at edge {edge}: full sigma != extended "
                "mu_e (frozen-row mutation mismatch)"
            )
        node = self.node_of_edge(edge)
        last = None
        for d in ("fwd", "inv"):
            try:
                new_key, iso = self.atlas.mutate(key, node, d)
                break
            except ValueError as exc:
                last = exc
        else:
            raise last
        mb2 = tri2.mutable_block()
        if self._node_exchange(new_key) != mb2:
            raise RuntimeError(
                f"bordered flip at edge {edge}: mutated chart's node-basis "
                "exchange != flipped mutable block"
            )
        self._tri[new_key] = tri2
        return new_key, iso

    def flip_path(self, edges, key=()) -> tuple:
        key = BPSAtlas._norm_key(key)
        for e in edges:
            key, _ = self.flip(e, key)
        return key

    # -- the native chart algebra ------------------------------------------
    def chart_spec(self, key) -> list[tuple]:
        A = self.atlas.chart(key)
        solver = _node_basis_solver([tuple(g) for g in A.node_charges])
        return [tuple(solver(tuple(g))) for g in A.spec]

    def chart_kalg(self, key=()) -> BPSKAlgebra:
        """The chart as a `BPSKAlgebra` on the flipped chart's own
        mutable block with the flip-transported spec (no auto-find)."""
        key = BPSAtlas._norm_key(key)
        if key not in self._native:
            mb = self._tri[key].mutable_block()
            n = len(mb)
            self._native[key] = BPSKAlgebra(
                pairing=mb,
                node_charges=[tuple(1 if j == i else 0 for j in range(n))
                              for i in range(n)],
                spec=self.chart_spec(key),
            )
        return self._native[key]

    def chart_iso(self, key=()) -> KAlgebraIso:
        """Certified `KAlgebraIso` root_kalg -> chart_kalg(key) (atlas
        transition composed with the node-basis recoordinatization)."""
        key = BPSAtlas._norm_key(key)
        native = self.chart_kalg(key)
        if not key:
            return KAlgebraIso.identity(native, name="bordered-atlas[root]")
        A = self.atlas.chart(key)
        atlas_iso = self.atlas.iso((), key)
        solver = _node_basis_solver([tuple(g) for g in A.node_charges])
        nodes = [tuple(int(x) for x in g) for g in A.node_charges]
        rank = len(nodes[0])

        def fwd(root_label):
            img = atlas_iso.map(Element({tuple(root_label): _ONE}))
            return Element({tuple(solver(tuple(l))): c
                            for l, c in img.terms.items()})

        def inv(nb_label):
            g = tuple(sum(int(c) * nodes[i][j] for i, c in enumerate(nb_label))
                      for j in range(rank))
            return atlas_iso.inverse(Element({g: _ONE}))

        return KAlgebraIso(self.root_kalg, native, fwd, inv,
                           name=f"bordered-atlas[{BPSAtlas._key_str(key)}]")


__all__ = ["SkeinSphereAtlas", "BorderedSkeinAtlas"]
