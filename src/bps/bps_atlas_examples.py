"""Complete `BPSAtlas` examples — the Argyres–Douglas zoo.

A **self-contained** gallery of *complete* atlases for the example
theories: each is built from a hard-coded BPS-chart literal `(pairing,
node_charges, spec)` — self-contained literals, with no dependencies
beyond the BPS realisation layer (`bps_kalgebra` + `bps_atlas`).

For a **finite-type** `A_𝖖[T]` the cluster exchange graph is finite, so the atlas
can be *completed*: every chart materialized (`BPSAtlas.complete()`).  All the
theories here complete in well under a second with a small chart graph
(`n_charts ≈ 2·rank`), so a complete atlas is cheap.  The unflavoured /
square-quiver members (A-even, D, E) additionally fold onto a fundamental domain
(`n_classes`); the flavoured `[A1,A_odd]` and the `sqed1` chart (frozen flavour
node) complete too but are not folded (the chart-iso recognition needs a
full-rank square quiver) — `classified=False`.

A second, sharper view is the **mutation-complete** folded atlas
(`BPSAtlas.mutation_complete()`): every node-mutation is folded by
chart-isomorphism (node-permutation + Γ-automorphism), so the pentagon comes
back as a **single chart with two outgoing mutation self-loops** (the mutation
of the pentagon's BPS quiver is isomorphic to itself), `[A1,A3]` as four charts,
`[A1,D4]` as ten, `[A1,E6]` as sixty-seven, `[A1,E8]` as one thousand five
hundred and seventy-four.  Raw charges grow without bound under mutation, but the
chart-iso classes are finite for these finite-type theories, so the fold
converges (`closed=True`).  **Every square example closes**
— the fold runs on the charge-space quiver mutation, so
there is no local-move budget to exhaust; only the non-square `sqed1` is
unfoldable.  Build cost grows with the chart count (sub-second through `[A1,E6]`,
~40s for `[A1,E8]`'s 1574 charts) but a built atlas is saved as data, so the
build is paid once.

    from bps_atlas_examples import atlas, complete_atlas, demo, EXAMPLE_NAMES
    At = atlas('pentagon')                 # a BPSAtlas seeded at the BPS chart
    At, rec = complete_atlas('pentagon')   # rec = {n_charts, n_classes, closed, ...}
    At, mc = mutation_complete_atlas('pentagon')  # mc = {n_charts:1, self_loops:2, ...}
    demo()                                 # build + complete every example, print tables
"""
from __future__ import annotations

from bps_kalgebra import BPSKAlgebra
from bps_atlas import BPSAtlas


# AD-zoo example charts: name -> (pairing, node_charges, spec, label).
# Self-contained BPS-chart literals — example theories whose complete
# atlas is small.
_DATA = {
    'sqed1': dict(
        label='U(1)+N_f=1 (SQED1)',
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(1, 0)],
        spec=[(1, 0)],
    ),
    'pentagon': dict(
        label='[A1,A2] Argyres-Douglas (pentagon)',
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(1, 0), (0, 1)],
        spec=[(1, 0), (0, 1)],
    ),
    'a3': dict(
        label='[A1,A3] = hexagon (u1-flavoured)',
        pairing=[[0, 1, 0], [-1, 0, 1], [0, -1, 0]],
        node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        spec=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    ),
    'a1d3': dict(
        label='[A1,D3]',
        pairing=[[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
        node_charges=[(1, 0, 0), (0, 1, 1), (0, 1, -1)],
        spec=[(1, 0, 0), (0, 1, 1), (0, 1, -1)],
    ),
    'heptagon': dict(
        label='[A1,A4] (heptagon)',
        pairing=[[0, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1], [0, 0, -1, 0]],
        node_charges=[(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
        spec=[(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
    ),
    'a1d4': dict(
        label='[A1,D4] (triality)',
        pairing=[[0, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 0]],
        node_charges=[(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 1), (0, 0, 1, -1)],
        spec=[(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, -1), (0, 0, 1, 1)],
    ),
    'a5': dict(
        label='[A1,A5] = octagon (u1-flavoured)',
        pairing=[[0, 1, 0, 0, 0], [-1, 0, 1, 0, 0], [0, -1, 0, 1, 0], [0, 0, -1, 0, 1], [0, 0, 0, -1, 0]],
        node_charges=[(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)],
        spec=[(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)],
    ),
    'a1d5': dict(
        label='[A1,D5]',
        pairing=[[0, 1, 0, 0, 0], [-1, 0, 1, 0, 0], [0, -1, 0, 1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]],
        node_charges=[(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 1), (0, 0, 0, 1, -1)],
        spec=[(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, -1), (0, 0, 0, 1, 1)],
    ),
    'nonagon': dict(
        label='[A1,A6] (nonagon)',
        pairing=[[0, 1, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 0], [0, -1, 0, 1, 0, 0], [0, 0, -1, 0, 1, 0], [0, 0, 0, -1, 0, 1], [0, 0, 0, 0, -1, 0]],
        node_charges=[(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1)],
        spec=[(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1)],
    ),
    'e6': dict(
        label='[A1,E6]',
        pairing=[[0, 1, 0, 0, 0, 0], [-1, 0, -1, 0, 0, 0], [0, 1, 0, 1, 0, 1], [0, 0, -1, 0, -1, 0], [0, 0, 0, 1, 0, 0], [0, 0, -1, 0, 0, 0]],
        node_charges=[(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1)],
        spec=[(1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 1, 0), (0, 0, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0)],
    ),
    'a1d6': dict(
        label='[A1,D6]',
        pairing=[[0, 1, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 0], [0, -1, 0, 1, 0, 0], [0, 0, -1, 0, 1, 0], [0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0]],
        node_charges=[(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 1), (0, 0, 0, 0, 1, -1)],
        spec=[(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, -1), (0, 0, 0, 0, 1, 1)],
    ),
    'a7': dict(
        label='[A1,A7] = decagon (u1-flavoured)',
        pairing=[[0, 1, 0, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0], [0, 0, -1, 0, 1, 0, 0], [0, 0, 0, -1, 0, 1, 0], [0, 0, 0, 0, -1, 0, 1], [0, 0, 0, 0, 0, -1, 0]],
        node_charges=[(1, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 1)],
        spec=[(1, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 1)],
    ),
    'a1d7': dict(
        label='[A1,D7]',
        pairing=[[0, 1, 0, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0], [0, 0, -1, 0, 1, 0, 0], [0, 0, 0, -1, 0, 1, 0], [0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
        node_charges=[(1, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1, 1), (0, 0, 0, 0, 0, 1, -1)],
        spec=[(1, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1, -1), (0, 0, 0, 0, 0, 1, 1)],
    ),
    'e7': dict(
        label='[A1,E7]',
        pairing=[[0, 1, 0, 0, 0, 0, 0], [-1, 0, -1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0], [0, 0, -1, 0, -1, 0, -1], [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
        node_charges=[(1, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 1)],
        spec=[(1, 0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 1, 0, 0, 0)],
    ),
    'a1d8': dict(
        label='[A1,D8]',
        pairing=[[0, 1, 0, 0, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0], [0, 0, -1, 0, 1, 0, 0, 0], [0, 0, 0, -1, 0, 1, 0, 0], [0, 0, 0, 0, -1, 0, 1, 0], [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
        node_charges=[(1, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 0, 1, 1), (0, 0, 0, 0, 0, 0, 1, -1)],
        spec=[(1, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 0, 1, -1), (0, 0, 0, 0, 0, 0, 1, 1)],
    ),
    'e8': dict(
        label='[A1,E8]',
        pairing=[[0, 1, 0, 0, 0, 0, 0, 0], [-1, 0, -1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0], [0, 0, -1, 0, -1, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, -1, 0, -1, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0]],
        node_charges=[(1, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 0, 1)],
        spec=[(1, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0)],
    ),
}

EXAMPLE_NAMES = list(_DATA)

# Every **square** example mutation-completes
# — the charge-mutation fold has no local-move budget to exhaust, so
# it always closes for a finite theory.  Only `sqed1` (a rank-1 gauge node
# carrying a frozen flavour direction → non-square) is unfoldable: the chart-iso
# recognizer (node-perm + Γ-automorphism) needs a full-rank square quiver.
#   This curated list is the **sub-second** showcase (folds, in charts):
#     pentagon→1, a3/a1d3→4, heptagon→6, a1d4→10, a5→19, a1d5→26, nonagon→49,
#     e6→67, a1d6→80.
# The heavier square members close too, just slower (build once):
#   a7→150 (~2s), a1d7→246 (~3s), e7→416 (~4s), a1d8→810 (~19s), e8→1574 (~40s).
MUTATION_COMPLETE_NAMES = [
    "pentagon", "a3", "a1d3", "heptagon", "a1d4", "a5", "a1d5",
    "nonagon", "e6", "a1d6",
]

# Every square example (all but the non-square `sqed1`) — the full foldable set,
# including the heavier A7/D7/E7/D8/E8.  `build_all_mutation_complete(names=
# MUTATION_COMPLETE_ALL)` builds them all (~70s — a full build, not a demo).
MUTATION_COMPLETE_ALL = [n for n in EXAMPLE_NAMES if n != "sqed1"]


def chart(name: str) -> BPSKAlgebra:
    """The BPS chart (`BPSKAlgebra`) of example `name`, from its stored literal.
    The coefficient (flavour) ring is auto-derived from the Dirac pairing's
    kernel, so flavoured theories (`a3`/`a5`/`a7`, the `D`-series) come back
    flavoured without any extra wiring."""
    d = _DATA[name]
    return BPSKAlgebra(pairing=[list(r) for r in d["pairing"]],
                       node_charges=[tuple(g) for g in d["node_charges"]],
                       spec=[tuple(s) for s in d["spec"]])


def atlas(name: str) -> BPSAtlas:
    """A `BPSAtlas` seeded at example `name`'s BPS chart."""
    return BPSAtlas(chart(name))


def complete_atlas(name: str, *, max_charts: int = 512):
    """Build example `name`'s atlas and **complete** it.  Returns `(atlas,
    record)` where `record` is the `BPSAtlas.complete()` result
    (`{n_charts, n_classes, classified, closed, keys}`)."""
    At = atlas(name)
    return At, At.complete(max_charts=max_charts)


def build_all_complete(*, max_charts: int = 512) -> list[dict]:
    """Complete every example; return a record per theory (`name`, `label`,
    `rank`, `n_charts`, `n_classes`, `classified`, `closed`)."""
    out = []
    for name in EXAMPLE_NAMES:
        At, rec = complete_atlas(name, max_charts=max_charts)
        out.append({
            "name": name,
            "label": _DATA[name]["label"],
            "rank": len(At.root.lattice.pairing),
            "n_charts": rec["n_charts"],
            "n_classes": rec["n_classes"],
            "classified": rec["classified"],
            "closed": rec["closed"],
        })
    return out


def mutation_complete_atlas(name: str, *, max_charts: int = 20000):
    """Build example `name`'s atlas and **mutation-complete** it:
    fold every node-mutation by chart-isomorphism (node-perm +
    Γ-automorphism), via the charge-space quiver mutation (always closes for a
    finite theory).  Returns `(atlas, record)` where `record` is the
    `BPSAtlas.mutation_complete()` result (`{n_charts, charts, edges,
    self_loops, rank_regular, classified, closed}`).  The pentagon (`pentagon`)
    comes back as a single chart with two outgoing mutation self-loops; `sqed1`
    (non-square) comes back `classified=False`, `closed=False`."""
    At = atlas(name)
    return At, At.mutation_complete(max_charts=max_charts)


def build_all_mutation_complete(names=None, *, max_charts: int = 20000) -> list[dict]:
    """Mutation-complete the curated sub-second square examples
    (`MUTATION_COMPLETE_NAMES` by default — through `[A1,E6]`); return a record
    per theory (`name`, `label`, `rank`, `n_charts`, `n_edges`, `self_loops`,
    `closed`).  Pass `names=MUTATION_COMPLETE_ALL` for the full foldable zoo
    (incl. A7/D7/E7/D8/E8, ~70s); `sqed1` is non-square and does not fold."""
    out = []
    for name in (MUTATION_COMPLETE_NAMES if names is None else names):
        At, rec = mutation_complete_atlas(name, max_charts=max_charts)
        out.append({
            "name": name,
            "label": _DATA[name]["label"],
            "rank": len(At.root.lattice.pairing),
            "n_charts": rec["n_charts"],
            "n_edges": len(rec["edges"]),
            "self_loops": rec["self_loops"],
            "closed": rec["closed"],
        })
    return out


def demo() -> None:
    """Build + complete every example and print the table."""
    print("=== complete BPSAtlas examples (Argyres-Douglas zoo) ===")
    print(f"  {'theory':10s} {'rank':>4s} {'charts':>6s} {'classes':>7s} "
          f"{'closed':>6s}  label")
    for r in build_all_complete():
        cls = "-" if r["n_classes"] is None else str(r["n_classes"])
        print(f"  {r['name']:10s} {r['rank']:>4d} {r['n_charts']:>6d} "
              f"{cls:>7s} {str(r['closed']):>6s}  {r['label']}")

    print("\n=== mutation-complete BPSAtlas examples (folded by chart-iso) ===")
    print(f"  {'theory':10s} {'rank':>4s} {'charts':>6s} {'edges':>6s} "
          f"{'loops':>6s} {'closed':>6s}  label")
    for r in build_all_mutation_complete():
        print(f"  {r['name']:10s} {r['rank']:>4d} {r['n_charts']:>6d} "
              f"{r['n_edges']:>6d} {r['self_loops']:>6d} "
              f"{str(r['closed']):>6s}  {r['label']}")
