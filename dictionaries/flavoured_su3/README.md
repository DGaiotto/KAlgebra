# The flavoured BPS-quiver dictionary — quivers with a manifest `SU(3)`

**Shard format 3 since 2026-09-28**.  This tier stores ONLY
the **strongly connected** quivers carrying one node orbit of size 3 (a manifest `SU(3)`),
through total unfolded arrow weight 10: **42 strongly connected flavoured quivers in 6.5 kB**: 41 with a covariant spec, 1 `crystalline_only@3`.

**Every other quiver with this orbit family is answered by composition**: acyclic
ones, cyclic ones that are not strongly connected, and disconnected ones.
`dictionary_loader.lookup_flavoured(B, orbit_size=…)` does it in three steps.
- It splits `B` into its strongly connected components.
- It takes a plain component's spec from the enumerated tier, and a component
  carrying orbits from the flavoured tier of the sizes it carries.
- It places each orbit of single-node components consecutively, since
  identical nodes lie in one component or are all single nodes.

The design is the design record.

**The records** read as the enumerated tier's:
- `green;axioms@5`: a negating sequence whose product was checked on the cone
  to degree 5;
- `axioms@5`: an order-search spec accepted to degree 5;
- `crystalline_only@3`: no covariant spec found; the content `Ω(γ, r)` is to
  depth 3, stored and recomputed in the stored labelling, since the content
  depends on the engine's order.

The shipped producer is kept as each entry's `source`.

**The format-2 tier it replaced**: every connected quiver with this orbit
family through weight 10, 143,947 quivers (21,096 stored), 657 kB.
- It was checked entry by entry against the composition
  (a probe in the source repository): every stored entry is
  answered, no spec is lost, and `S` is reproduced on every sampled pair.
- Its census is kept in the manifest (`census_of_every_flavoured_quiver`).
- The sections below describe that retired tier, and git history keeps its
  shards.

## The retired format-2 tier (the record)

The `SU(3)` companion of `dictionaries/flavoured/`. **Read that tier's README
first** — the construction, the shard format, the acyclic coding and the checks
are identical, and only the flavour group differs. This file records what is
specific to `SU(3)`.

## What an entry is, and why the selector is the orbit size

An entry is a connected BPS quiver carrying a **node orbit of size 3**: three
nodes that pairwise do not pair (`B[i][j] = 0`) and every permutation of which
is an automorphism of the exchange matrix — three *identical, interchangeable*
nodes, so the theory has a manifest `SU(3)` rotating them (the author's *"as soon
as you have `N` identical nodes in the quiver, the theory has manifest `SU(N)`
symmetry"*, 2026-09-13).

**This is not "three doublets".** The `SU(2)` tier selects quivers with one node
*doublet* — a pair — and three mutually identical nodes contain **three
overlapping doublet pairs**, so a selector counting pairs would have put an
`SU(3)` quiver in a "three doublets" bucket. Three *disjoint* pairs is a
different object again — `SU(2)³`, one orbit of size 2 at each of three places.
So the selector is `node_orbits(B, size)` with an explicit size, and the shipped
`SU(2)` tier is exactly its `size = 2` case (asserted in
the suite in the source repository).

The anchor is the **three-leaf quiver** — three identical nodes attached to one
centre — whose fold is the two-node flavoured quiver carrying the `SU(3)`
fundamental at the orbit node, `factors = (1, 3)`.

## What is shipped

`orbit_size = 3`, one orbit. **Complete through total arrow weight `e ≤ 10`**,
the weight being the **unfolded** quiver's as always: **143,947 quivers, 657 KB
gzipped over 31 shards**.

| certification | entries |
|---|---:|
| `covariant_spec:strip` (acyclic; **coded**, not stored) | 122,851 |
| `covariant_spec:blocks` (block-mutation bidirectional BFS) | 21,057 |
| `crystalline_only@3` (no covariant spec found within budget) | 39 |
| `crystalline_failed@D` (`Ω` not a character — a conjecture counterexample) | **0** |

So 143,908 of 143,947 carry an `SU(3)`-covariant spec, and **no entry fails to
have `Ω` a `G`-character** — the emergent claim of `notes/flavoured_s_axioms.md` §3,
now also at `SU(3)`, where `Ω` takes values in `R(SU(3)) ⊗ P` and the derived
reps reach the adjoint, whose zero weight has multiplicity two. That is the case
the `R(G)`-valued bookkeeping exists for: a rep with a repeated weight cannot sit
at a node of an unfolded quiver, so it is not reachable by unfolding.

Two differences from the `SU(2)` tier worth noting:

* **It is much smaller at the same weight** — 143,947 against 1,022,925 — because
  three mutually identical nodes are a far stronger condition than two. Unfolding
  an orbit of size `N` at a node of degree `d` adds `(N−1)·d` to the weight, so
  an `SU(3)` entry of weight `≤ E` folds to weight `≤ E − 2` rather than `E − 1`.
* **The crystalline rate is lower**, 39 in 21,096 stored (0.18 %) against 666 in
  200,173 (0.33 %).

## How it is checked

Same two routes as the `SU(2)` tier, and the same discipline: every spec is
rebuilt against `S` on a cone **strictly wider** than the search's, since an
in-cone rebuild is a screen and not a certificate (the audit).

    PYTHONPATH=. python a probe in the source repository \
        --dir dictionaries/flavoured_su3 --cutoff 4

Measured 2026-09-22 on Perimeter's Symmetry cluster, where weight 10 was built:
**18,785 / 18,785 weight-10 stored entries at depth 5, 0 failures** (258 s); and
2026-09-16, **2,311 / 2,311 at depth 4, 0 failures** (332 s, `--cutoff 3`) for
the earlier weights. The census also checks out cell by cell against
`manifest.json`.

⚠ The **refinement statement is checked only where the unflavoured tier
reaches**: `dictionaries/enumerated/` ships to `E = 9`, so the membership check
ran on the 2,311 entries of weight `≤ 9`, not on the 18,785 of weight 10.

## Building more

    PYTHONPATH=. python dictionaries/build_flavoured_by_rank.py \
        --orbit-size 3 --max-weight 11 --workers N --ship <staging dir>

Cells already present are skipped, so this resumes. Build to a **staging
directory**, never into the tracked tier — an in-progress build leaves the tier
in a state its manifest does not describe. On a batch scheduler add `--shard
i/K` with `--merge-shards`: the output is byte for byte the plain builder's. See
the design record
for the rest of the operational detail; it is written for `SU(2)` but every part
except the cost curve applies verbatim.

**Tracked in git**, like every permanent tier, at 657 KB gzipped — see
the design notes "Workflow" for the standing ruling on `dictionaries/` artifacts.
