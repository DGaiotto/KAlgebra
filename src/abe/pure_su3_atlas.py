"""`pure_su3_atlas` — a `BPSAtlas` for **pure SU(3)** super-Yang–Mills.

Pure SU(3) was the one corner the gauge catalogue
explicitly **deferred** — "cyclic 4-node ... deferred rather than fabricate a
quiver/spec".  It is built here from the **closed-form** pure-gauge construction
already in the repo, `pure_ade.PureADE([("A", 2)])` (the bipartite-Coxeter MGS on
the doubled A₂ Dynkin quiver — no BPS search), so there is nothing to fabricate:

    BPS quiver (Dirac pairing nodes·B·nodesᵀ)      strong-coupling spec
        [ 0  2  0 -1]                               6 hypers
        [-2  0  1  0]    = two Kronecker-2 pairs    (= r·h = 2·3 for A₂)
        [ 0 -1  0  2]      (0↔1, 2↔3) joined in
        [ 1  0 -2  0]      a 4-cycle by single arrows

The seed chart is the **strong-coupling chamber** (finite spec, length 6),
unflavoured (`R = Z`, pure gauge).  The atlas's canonical necklace rotation
closes with **period 12** and **monodromy = ρ²** (verified on the node labels);
ρ² itself *drifts* the charges, so — like every asymptotically-free gauge theory
— **ρ is infinite-order** (the Witten effect; user-confirmed 2026-06-28).

### The wild-chamber boundary (the careful part — user, 2026-06-28)

> *"some mutations bring you to wild chambers likely with no spec."*

Pure SU(3) for the first time in this catalogue has genuinely **wild** BPS
chambers: mutating off the necklace-rotation orbit reaches chambers with an
**infinite BPS spectrum and no finite spectrum generator**.  Concretely:

  * the spec-based chart graph **declines** those moves at `max_local_moves=0`
    ("does not cooperate") — so `BPSAtlas.mutate` / `folded_graph` stay inside the
    finite chamber by construction and never hang;
  * a raw quiver mutation into a wild chamber has **no finite negating sequence**
    (`BPSQuiver.find_negating_sequence` returns `None`, confirmed to depth 22);
  * the spec-free direct-`S` engine does **not σ-stabilize** there — built with a
    fixed cone cutoff it yields a *cutoff-dependent, non-convergent* `multiply`
    (more terms / growing negative q-powers as the cutoff grows), so it must not
    be trusted in a wild chamber.

So the pure-SU(3) atlas = the finite strong-coupling chamber + its certified
rotation orbit; the wild chambers are **detected and excluded**, not approximated.

Public API:
  * `pure_su3_chart()         -> BPSKAlgebra`   (the strong-coupling seed)
  * `pure_su3_atlas(**kw)     -> BPSAtlas`
  * `pure_su3_quiver()        -> BPSQuiver`     (raw quiver, for chamber probes)
  * `mutate_quiver(path)      -> BPSQuiver`     (raw mutation, leaves the necklace orbit)
  * `chamber_has_finite_spec(quiver, *, spec_depth=18) -> list[int]|None`
  * `pure_su3_chamber_census(*, max_depth=3, spec_depth=14) -> dict`
  * `WILD_CHAMBER_PATHS`      -> known wild raw-mutation paths (depth ≤ 2)
  * `pure_su3_spec_free_chart(nodes, cutoff) -> BPSKAlgebra`  (the unreliable wild build)
"""
from __future__ import annotations

import os
import sys
from collections import deque
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pure_ade as _pa
from bps_kalgebra import BPSKAlgebra
from bps_atlas import BPSAtlas
from bps_quiver_tools import BPSQuiver


__all__ = [
    "pure_su3_pure_ade",
    "pure_su3_chart",
    "pure_su3_atlas",
    "pure_su3_quiver_pairing",
    "pure_su3_quiver",
    "mutate_quiver",
    "chamber_has_finite_spec",
    "pure_su3_chamber_census",
    "WILD_CHAMBER_PATHS",
    "pure_su3_spec_free_chart",
]


# ---------------------------------------------------------------------------
# the closed-form pure-SU(3) BPS data  ->  canonical BPSKAlgebra
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def pure_su3_pure_ade():
    """The `pure_ade.PureADE([("A", 2)])` builder for pure SU(3)."""
    return _pa.PureADE([("A", 2)])


def pure_su3_chart() -> BPSKAlgebra:
    """The strong-coupling `BPSKAlgebra` chart of pure SU(3).

    Rank 4, unflavoured (`TrivialZPlusRing`), finite spec of length 6.  The spec
    is recipe-valid by construction (`PureADE` runs `verify_spectrum_generator`),
    so the chart is built with `verify="off"`.  (No `cone_witness` is passed — it
    does not affect `multiply`/`ρ` for this chart.)"""
    P = pure_su3_pure_ade()
    return BPSKAlgebra(
        pairing=[list(r) for r in P.B],
        node_charges=[tuple(g) for g in P.nodes],
        spec=[tuple(g) for g in P.spec],
        verify="off",
    )


def pure_su3_atlas(**kwargs) -> BPSAtlas:
    """A `BPSAtlas` seeded at the pure-SU(3) strong-coupling chart.

    `kwargs` are forwarded to `BPSAtlas` (e.g. `build_S_charts`,
    `spec_free_sigma`); the default seed is spec-mode (finite chamber)."""
    return BPSAtlas(pure_su3_chart(), **kwargs)


# ---------------------------------------------------------------------------
# raw quiver + chamber census  (the wild-chamber map)
# ---------------------------------------------------------------------------
def pure_su3_quiver_pairing():
    """`(nodes, B)` of the pure-SU(3) charge lattice (Γ_sc coords), as consumed by
    `BPSQuiver.from_pairing`.  The BPS-quiver Dirac pairing is `nodes·B·nodesᵀ`."""
    P = pure_su3_pure_ade()
    return [tuple(g) for g in P.nodes], [list(r) for r in P.B]


def pure_su3_quiver() -> BPSQuiver:
    """The raw pure-SU(3) `BPSQuiver` (strong-coupling representative)."""
    nodes, B = pure_su3_quiver_pairing()
    return BPSQuiver.from_pairing(nodes, B)


def mutate_quiver(path) -> BPSQuiver:
    """The `BPSQuiver` reached by the raw mutation `path` (a sequence of node
    indices) from the strong-coupling representative.  Unlike `BPSAtlas.mutate`
    (which necklaces the spec and stays in the finite chamber), this mutates the
    quiver directly and **can leave the finite chamber into a wild one**."""
    Q = pure_su3_quiver()
    for k in path:
        Q = Q.mutate(int(k))
    return Q


def chamber_has_finite_spec(quiver: BPSQuiver, *, spec_depth: int = 18):
    """A finite negating sequence for `quiver` (⇒ a finite spectrum generator),
    or `None` if none is found within `spec_depth` (⇒ a **wild chamber**, no
    finite spec).  Thin wrapper over `BPSQuiver.find_negating_sequence`."""
    return quiver.find_negating_sequence(max_depth=spec_depth)


def pure_su3_chamber_census(*, max_depth: int = 3, spec_depth: int = 14) -> dict:
    """Census the mutation classes within `max_depth` raw mutations (deduped up
    to node relabelling by the sorted charge multiset), classifying each as
    **finite-spec** or **wild** (no finite negating sequence within `spec_depth`).

    Returns `{n_chambers, n_finite, n_wild, finite_paths, wild_paths}` — the
    explicit map of where the finite strong-coupling chambers sit and where the
    wild chambers begin (already at depth 2 for pure SU(3))."""
    root = pure_su3_quiver()

    def key(Q):
        return tuple(sorted(tuple(c) for c in Q.charges))

    seen = {key(root): (root, [])}
    frontier = deque([(root, [])])
    while frontier:
        Q, path = frontier.popleft()
        if len(path) >= max_depth:
            continue
        for k in range(4):
            Q2 = Q.mutate(k)
            kk = key(Q2)
            if kk in seen:
                continue
            seen[kk] = (Q2, path + [k])
            frontier.append((Q2, path + [k]))

    finite_paths, wild_paths = [], []
    for _, (Q, path) in seen.items():
        ns = chamber_has_finite_spec(Q, spec_depth=spec_depth)
        (finite_paths if ns is not None else wild_paths).append(tuple(path))
    return {
        "n_chambers": len(seen),
        "n_finite": len(finite_paths),
        "n_wild": len(wild_paths),
        "finite_paths": sorted(finite_paths),
        "wild_paths": sorted(wild_paths),
    }


# A few raw-mutation paths that land in wild chambers (no finite spec), already
# at depth 2 — the concrete demonstration of the author's warning.  Verified by
# `pure_su3_chamber_census` (find_negating_sequence -> None even at depth 22).
WILD_CHAMBER_PATHS = (
    (0, 1),
    (1, 0),
    (2, 3),
    (3, 2),
)


def pure_su3_spec_free_chart(nodes, cutoff: int) -> BPSKAlgebra:
    """A **spec-free** `BPSKAlgebra` on the quiver `nodes`, built with the
    direct-`S` engine at a *fixed* cone `cutoff` and the axiom-derived σ
    (`spec_free_sigma="principled"`).

    In a wild chamber this build does **not** σ-stabilize, so its `multiply` is
    cutoff-dependent and **non-convergent** — use only to *demonstrate* that
    unreliability, never as ground truth (see the module docstring)."""
    _, B = pure_su3_quiver_pairing()
    return BPSKAlgebra(
        pairing=B,
        node_charges=[tuple(n) for n in nodes],
        build_S=True,
        spec_free_sigma="principled",
        build_S_cutoff=int(cutoff),
    )
