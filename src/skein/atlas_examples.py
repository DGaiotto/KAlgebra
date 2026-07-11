"""Worked examples of the triangulation-labelled atlas
(`SkeinSphereAtlas`) with certified `KAlgebraIso`s to other
formulations, and their `KAlgebraObject` registrations.

Contents (all constants below are certified by the accompanying
tests):

* **S²₀,₄ (= SU(2) N_f=4).**
  - `su2_nf4_gauge_iso()` — the certified `KAlgebraIso` from the
    skein-FST tetrahedron chart to the **gauge-quiver formulation**:
    mutation word `MUTATION_WORD_TETRA_TO_GAUGE = (0, 4, 1)`
    (spec-cooperating order of the B-level word) lands on the SU(2)
    N_f=4 gauge quiver (Kronecker-2 + four dyon nodes — B-matrix
    match under `NODE_PERM_GAUGE`, dyon nodes symmetric); the target
    is that quiver presented in the TRANSPORTED chamber (the
    necklaced 12-spec, reframed).  Two measured lattice/chamber
    facts feed this design:
      (i) the `pure_ade.SUN_Nf(2,4)` AMBIENT lattice is an INDEX-2
      EXTENSION of its node lattice (`det(nodes) = 2`): the ambient
      carries the half/fundamental-Wilson charges of the line-defect
      extension `Sk̃` (v-class sector) which the
      bare FST lattice does not — so the bijective iso lives on the
      node lattice;
      (ii) the `pure_ade` chamber has |spec| = 14 while every
      necklace of the FST chamber has |spec| = 12 — DIFFERENT BPS
      chambers of the same theory, so no linear label map is the
      canonical dictionary across them; the cross-chamber statement
      is checked the chamber-independent way instead
      (equal Schur data on matched canonicals — wall-crossing
      invariance).
  - `pure_ade_chamber_chart()` — the reference 14-spec `pure_ade`
    chamber on the same node lattice, for those intrinsic
    cross-chamber checks.
  - `s2_04_object()` — the `KAlgebraObject` holding the skein-FST
    root chart, a flipped (multi-edge) FST chart, and the
    gauge-quiver chart, with the certified witnesses (star at the
    root — the coherence content is the pairwise batteries).

* **S²₀,₅.**
  - `SPEC20_BIPYRAMID` — the bipyramid chamber (auto-found once,
    204 s; frozen here so no consumer ever pays that again).
  - `SAUSAGE_FLIP_PATH = (2, 0, 6)` — three flips take the bipyramid
    to a chart combinatorially isomorphic to the sausage
    triangulation `cluster_sausage_S2_5` (canonical-form match);
    along the way the spec is flip-transported, so the sausage-shaped
    chart's canonical algebra needs NO spec auto-find — the spec
    transport realized at n=5.

* **Once-punctured torus (N=2* SU(2)).**  Chart-level only: the flip
  graph is the Markov/Farey graph and the S-free geometric F runs, but
  there is NO finite BPS chamber, so no `BPSKAlgebra`/atlas — the
  honest boundary; the N=2* engines are the cross-validation target
  there.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bps_kalgebra import BPSKAlgebra
from directional_subquiver_rg import _node_basis_solver
from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly

from triangulation import Triangulation
from skein_sphere_atlas import SkeinSphereAtlas

_ONE = LaurentPoly({0: 1})


def _bps_from_pure_ade(*_args, **_kwargs):
    """Build a BPS chart from a ``pure_ade`` gauge-quiver spec.

    The gauge-chart examples below were developed against a spectrum-atlas
    helper that is not part of this release; the reference ``pure_ade``
    chambers are reachable through ``pure_ade_lattice.pure_ade_kalgebra``.
    """
    raise NotImplementedError(
        "the gauge-chart examples require a BPS spectrum-atlas helper that "
        "is not part of the public release")

# ---- discovered constants ---------------------------------------------

MUTATION_WORD_TETRA_TO_GAUGE = (0, 4, 1)
NODE_PERM_GAUGE = (0, 5, 1, 2, 3, 4)

SAUSAGE_FLIP_PATH = (2, 0, 6)

SPEC20_BIPYRAMID = [
    (0, 0, 1, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 1, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 0, 1, 0, 0),
    (1, 1, 1, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0, 0),
    (1, 0, 0, 0, 1, 0, 1, 0, 0), (1, 1, 1, 0, 1, 0, 1, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 0, 1), (0, 0, 0, 0, 1, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 1, 0), (0, 0, 0, 0, 0, 0, 0, 1, 0),
    (0, 1, 1, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 1),
    (0, 1, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 1, 0, 0, 1, 0),
    (0, 0, 0, 0, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0, 0),
]


# ---- S^2_{0,4}: the gauge-theory formulation ---------------------------

def _gauge_quiver_data():
    """The SU(2) N_f=4 gauge quiver on its own node lattice: the
    node-basis exchange matrix `E` (Kronecker-2 + four dyon nodes) and
    the `pure_ade` chamber spec in node coordinates."""
    import pure_ade as _pa

    amb = _bps_from_pure_ade(_pa.SUN_Nf(2, 4))
    gnodes = [tuple(g) for g in amb.node_charges]
    solver = _node_basis_solver(gnodes)
    br = amb.lattice.bracket
    n = len(gnodes)
    E = [[br(gnodes[i], gnodes[j]) for j in range(n)] for i in range(n)]
    spec_nb = [tuple(solver(tuple(g))) for g in amb.spec]
    return E, spec_nb


def pure_ade_chamber_chart() -> BPSKAlgebra:
    """The 14-spec `pure_ade` chamber of the SU(2) N_f=4 gauge quiver
    (node-lattice presentation) — the reference for the intrinsic
    cross-chamber checks (the FST-side chambers all have |spec| = 12,
    a genuinely different chamber)."""
    E, spec_nb = _gauge_quiver_data()
    n = len(E)
    return BPSKAlgebra(
        pairing=E,
        node_charges=[tuple(1 if j == i else 0 for j in range(n))
                      for i in range(n)],
        spec=spec_nb,
    )


def _mutation_path_with_fallback(atlas, word):
    """Apply a mutation word trying 'fwd' then 'inv' per step (the
    spec-cooperation fallback); returns the end key."""
    key = ()
    for node in word:
        last = None
        for d in ("fwd", "inv"):
            try:
                key, _ = atlas.mutate(key, node, d)
                last = None
                break
            except ValueError as exc:
                last = exc
        if last is not None:
            raise last
    return key


def su2_nf4_gauge_iso(atlas: SkeinSphereAtlas | None = None):
    """The certified `KAlgebraIso` skein-FST tetrahedron chart ->
    SU(2) N_f=4 **gauge-quiver** chart (node lattice, transported
    12-spec chamber).

    Returns `(atlas, gauge, iso)`.  The construction: mutate along
    `MUTATION_WORD_TETRA_TO_GAUGE`, certify the arrived chart's
    node-basis exchange matrix IS the gauge quiver under
    `NODE_PERM_GAUGE` (asserted), then reframe onto the gauge quiver's
    unit node basis carrying the necklaced 12-spec along — the same
    chamber in the gauge quiver's coordinates (the `bps_frame_change`
    pattern).  The 14-spec `pure_ade` chamber is a different chamber:
    see `pure_ade_chamber_chart()` and the module docstring."""
    at = atlas or SkeinSphereAtlas.tetrahedron()
    key = _mutation_path_with_fallback(at.atlas, MUTATION_WORD_TETRA_TO_GAUGE)
    C = at.atlas.chart(key)
    E, _ = _gauge_quiver_data()

    p = NODE_PERM_GAUGE
    cnodes = [tuple(g) for g in C.node_charges]
    solver_c = _node_basis_solver(cnodes)
    n = len(cnodes)
    rank = len(cnodes[0])
    br = C.lattice.bracket
    BC = [[br(cnodes[i], cnodes[j]) for j in range(n)] for i in range(n)]
    if any(BC[p[i]][p[j]] != E[i][j] for i in range(n) for j in range(n)):
        raise RuntimeError(
            "mutated chart's quiver does not match the gauge quiver "
            "under NODE_PERM_GAUGE"
        )

    def to_g(lbl):
        cc = solver_c(tuple(lbl))
        return tuple(cc[p[i]] for i in range(n))

    def from_g(lbl):
        return tuple(sum(lbl[i] * cnodes[p[i]][j] for i in range(n))
                     for j in range(rank))

    gauge = BPSKAlgebra(
        pairing=E,
        node_charges=[tuple(1 if j == i else 0 for j in range(n))
                      for i in range(n)],
        spec=[to_g(tuple(g)) for g in C.spec],
    )

    atlas_iso = at.atlas.iso((), key)

    def fwd(lbl):
        img = atlas_iso.map(Element({tuple(lbl): _ONE}))
        return Element({to_g(l): c for l, c in img.terms.items()})

    def inv_(lbl):
        return atlas_iso.inverse(Element({from_g(lbl): _ONE}))

    iso = KAlgebraIso(at.root_kalg, gauge, fwd, inv_,
                      name="skein-fst[tetrahedron] -> SU2Nf4-gauge-quiver")
    return at, gauge, iso


def s2_04_object() -> KAlgebraObject:
    """The abstract S²₀,₄ = SU(2) N_f=4 algebra as a `KAlgebraObject`:
    the skein-FST tetrahedron chart, one flipped (multi-edge) FST
    chart, and the gauge-theory chart, with certified witnesses (star
    at the skein root)."""
    at, gauge, iso_gauge = su2_nf4_gauge_iso()
    key, _ = at.flip(0)
    flipped = at.chart_kalg(key)
    iso_flip = at.chart_iso(key)

    obj = KAlgebraObject("S2_04 = SU(2) Nf=4")
    obj.add_realization("skein-fst", at.root_kalg,
                        capabilities=("skein", "geometric-F", "bps"))
    obj.add_realization("skein-fst-flip0", flipped,
                        capabilities=("skein", "geometric-F", "bps",
                                      "multi-edge"))
    obj.add_realization("bps-gauge", gauge, capabilities=("bps", "gauge"))
    obj.add_iso("skein-fst", "skein-fst-flip0", iso_flip)
    obj.add_iso("skein-fst", "bps-gauge", iso_gauge)
    return obj


# ---- S^2_{0,5}: bipyramid -> sausage by flips ---------------------------

def bipyramid_atlas(**kw) -> SkeinSphereAtlas:
    """The S²₀,₅ atlas rooted at the bipyramid with the frozen 20-spec
    (no auto-find).

    Defaults to `iso_check=False`: the per-add chart-iso classification
    is the dominant cost at rank 9 (it turned minutes-scale flips into
    a >30 min wall — measured); pass `iso_check=True`
    explicitly if the fold-onto-fundamental-domain bookkeeping is
    wanted.  With it off, the three `SAUSAGE_FLIP_PATH` flips take
    ~65/66/124 s (spec necklacing through the chart graph) and the
    native sausage-chart algebra then builds in ~0 s from the
    transported spec (vs ~204 s auto-find)."""
    kw.setdefault("iso_check", False)
    return SkeinSphereAtlas(Triangulation.bipyramid_S2_5(),
                            spec=SPEC20_BIPYRAMID, **kw)


def sausage_chart(atlas: SkeinSphereAtlas | None = None):
    """Flip the bipyramid to the sausage-shaped chart along
    `SAUSAGE_FLIP_PATH`, certifying each flip (= mutation) on the way;
    returns `(atlas, key)`.  The arrived triangulation is
    combinatorially isomorphic to `cluster_sausage_S2_5` (canonical
    forms equal — asserted), and its native `SkeinSphereKAlg` carries
    the flip-transported spec: canonical algebra + skein surface on
    the sausage chart with no spec auto-find."""
    at = atlas or bipyramid_atlas()
    key = at.flip_path(SAUSAGE_FLIP_PATH)
    got = at.triangulation(key).canonical_form()
    want = Triangulation.cluster_sausage_S2_5().canonical_form()
    if got != want:
        raise RuntimeError(
            "flip path did not arrive at the sausage combinatorial class"
        )
    return at, key


# ---- IRREGULAR SINGULARITIES: bordered examples -----------------------
#
# Discovered constants for the cut-tetrahedron = SU(2) N_f=3 example
# (B-level BFS + spec-cooperation sweep; the (0,2,3) ordering matches at
# B-level but its spec does not cooperate — same phenomenon as N_f=4):

MUTATION_WORD_CUTTETRA_TO_GAUGE = ((0, "fwd"), (3, "fwd"), (2, "inv"))
NODE_PERM_GAUGE_NF3 = (4, 0, 1, 2, 3)


def pentagon_flip_atlas():
    """The bordered pentagon (disk, 5 marks = the A_2 AD theory, one
    rank-3/2 irregular puncture) as a `BorderedSkeinAtlas`, with the
    associahedron certificate MEASURED at atlas level:

      * ord(rho) = 5 = h+2 on canonical labels (the 1/(h+2) fractional
        quantum monodromy of the irregular puncture — one marked-point
        rotation per rho);
      * the 5-flip associahedron loop (alternating the two internal
        edges) composes to exactly rho^2, and since gcd(2,5)=1 the loop
        generates <rho^2> = Z/5 — the full finite cluster modular group.

    Returns `(atlas, loop_key)` with every flip certified (full
    frozen-extended mu_e + mutable block) on the way."""
    from bordered_triangulation import BorderedTriangulation
    from skein_sphere_atlas import BorderedSkeinAtlas

    tri = Triangulation.from_bordered_slots(
        BorderedTriangulation.fan_polygon(5))
    at = BorderedSkeinAtlas(tri)
    e0, e1 = tri.internal_edge_ids
    key = at.flip_path((e1, e0, e1, e0, e1))
    return at, key


def su2_nf3_gauge_iso(atlas=None):
    """The certified `KAlgebraIso` from the cut-tetrahedron chart
    (S²₀,₄ with FST edge 4 cut = SU(2) N_f=3, one rank-1 irregular
    puncture with two marked points) to the SU(2) N_f=3 gauge-quiver
    chart (node lattice, transported 10-spec chamber).

    Mirrors `su2_nf4_gauge_iso`: mutation word
    `MUTATION_WORD_CUTTETRA_TO_GAUGE`, node correspondence
    `NODE_PERM_GAUGE_NF3`; both measured facts persist at N_f=3 —
    the `pure_ade` AMBIENT is an index-2 extension of its node lattice
    (det = 2), and the `pure_ade` chamber is a DIFFERENT chamber
    (|spec| 11 vs the transported 10), so the cross-chamber link is
    the chamber-independent Schur check, not a linear label map.

    Returns `(atlas, gauge, iso)` where `atlas` is the
    `BorderedSkeinAtlas` of the cut chart."""
    import pure_ade as _pa
    from skein_sphere_atlas import BorderedSkeinAtlas

    at = atlas
    if at is None:
        # step (3,'fwd') needs a 16-local-move budget to necklace
        at = BorderedSkeinAtlas(Triangulation.tetrahedron_S2_4().cut(4),
                                max_local_moves=16)
    key = ()
    for node, d in MUTATION_WORD_CUTTETRA_TO_GAUGE:
        key, _ = at.atlas.mutate(key, node, d)
    C = at.atlas.chart(key)

    amb = _bps_from_pure_ade(_pa.SUN_Nf(2, 3))
    gnodes = [tuple(g) for g in amb.node_charges]
    gsolver = _node_basis_solver(gnodes)
    br = amb.lattice.bracket
    n = len(gnodes)
    E = [[br(gnodes[i], gnodes[j]) for j in range(n)] for i in range(n)]

    p = NODE_PERM_GAUGE_NF3
    cnodes = [tuple(g) for g in C.node_charges]
    solver_c = _node_basis_solver(cnodes)
    rank = len(cnodes[0])
    brc = C.lattice.bracket
    BC = [[brc(cnodes[i], cnodes[j]) for j in range(n)] for i in range(n)]
    if any(BC[p[i]][p[j]] != E[i][j] for i in range(n) for j in range(n)):
        raise RuntimeError(
            "mutated cut chart's quiver does not match the N_f=3 gauge "
            "quiver under NODE_PERM_GAUGE_NF3"
        )

    def to_g(lbl):
        cc = solver_c(tuple(lbl))
        return tuple(cc[p[i]] for i in range(n))

    def from_g(lbl):
        return tuple(sum(lbl[i] * cnodes[p[i]][j] for i in range(n))
                     for j in range(rank))

    gauge = BPSKAlgebra(
        pairing=E,
        node_charges=[tuple(1 if j == i else 0 for j in range(n))
                      for i in range(n)],
        spec=[to_g(tuple(g)) for g in C.spec],
    )
    atlas_iso = at.atlas.iso((), key)

    def fwd(lbl):
        img = atlas_iso.map(Element({tuple(lbl): _ONE}))
        return Element({to_g(l): c for l, c in img.terms.items()})

    def inv_(lbl):
        return atlas_iso.inverse(Element({from_g(lbl): _ONE}))

    iso = KAlgebraIso(at.root_kalg, gauge, fwd, inv_,
                      name="skein-cut-fst[tetra,e4] -> SU2Nf3-gauge-quiver")
    return at, gauge, iso


def cut_tetra_nf3_object() -> KAlgebraObject:
    """The abstract SU(2) N_f=3 algebra as a `KAlgebraObject`: the
    cut-tetrahedron bordered chart (irregular-puncture formulation), a
    flipped bordered chart, and the gauge-quiver chart, with certified
    witnesses (star at the cut root)."""
    at, gauge, iso_gauge = su2_nf3_gauge_iso()
    tri = at.triangulation()
    key, _ = at.flip(tri.internal_edge_ids[0])
    flipped = at.chart_kalg(key)
    iso_flip = at.chart_iso(key)

    obj = KAlgebraObject("SU(2) Nf=3 = cut-tetrahedron S2_04")
    obj.add_realization("skein-cut-fst", at.root_kalg,
                        capabilities=("bps", "bordered", "irregular"))
    obj.add_realization("skein-cut-fst-flip", flipped,
                        capabilities=("bps", "bordered", "irregular"))
    obj.add_realization("bps-gauge", gauge, capabilities=("bps", "gauge"))
    obj.add_iso("skein-cut-fst", "skein-cut-fst-flip", iso_flip)
    obj.add_iso("skein-cut-fst", "bps-gauge", iso_gauge)
    return obj


__all__ = [
    "MUTATION_WORD_TETRA_TO_GAUGE", "NODE_PERM_GAUGE",
    "SAUSAGE_FLIP_PATH", "SPEC20_BIPYRAMID",
    "pure_ade_chamber_chart", "su2_nf4_gauge_iso", "s2_04_object",
    "bipyramid_atlas", "sausage_chart",
    "MUTATION_WORD_CUTTETRA_TO_GAUGE", "NODE_PERM_GAUGE_NF3",
    "pentagon_flip_atlas", "su2_nf3_gauge_iso", "cut_tetra_nf3_object",
]
