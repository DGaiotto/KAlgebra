# The flavoured BPS-quiver dictionary — quivers with a manifest `SU(4) x SU(2)`

**Shard format 3 since 2026-09-28**.  This tier stores ONLY
the **strongly connected** quivers carrying a family of disjoint node orbits of sizes 4, 2 (a manifest `SU(4)×SU(2)`),
through total unfolded arrow weight 11: **no entry at all**: no strongly connected quiver of this weight carries such a family, so every one of its quivers is answered by composition.

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
family through weight 11, 24,064 quivers (2,036 stored), 71 kB.
- It was checked entry by entry against the composition
  (a probe in the source repository): every stored entry is
  answered, no spec is lost, and `S` is reproduced on every sampled pair.
- Its census is kept in the manifest (`census_of_every_flavoured_quiver`).
- The sections below describe that retired tier, and git history keeps its
  shards.

## The retired format-2 tier (the record)

The `SU(4) x SU(2)` member of the flavoured family. **Read
`dictionaries/flavoured/README.md` first** — the construction, the shard format,
the acyclic coding and the checks are identical across the family and only the
flavour group differs.

## What an entry is

A connected BPS quiver carrying **a quadruple and a disjoint pair of identical
nodes**: two orbits, of sizes 4 and 2, each a set of nodes that pairwise do not
pair (`B[i][j] = 0`) and every permutation of which is an automorphism of the
exchange matrix, and the two orbits sharing no node.

The acceptance is that `node_orbit_families(B, [4, 2])` has **exactly one**
element.  That is what keeps the degenerate neighbours out: six mutually
identical nodes split into a quadruple and a pair in fifteen ways, so such a
quiver is rejected here rather than filed as this tier's shape.

## What is shipped

`--orbit-sizes 4,2`, one family.  **Complete through total arrow weight
`e <= 11`**, the weight being the **unfolded** quiver's: **24,064 quivers,
71 kB gzipped over 19 shards** (the weight-11 cells on Perimeter's Symmetry
cluster, 24 workers on one AMD node; the earlier weights in 5 h 34 m, 1 worker,
1.15 GB peak).

| certification | entries |
|---|---:|
| `covariant_spec:blocks` | 2,036 |
| `covariant_spec:strip (coded, not stored)` | 22,028 |

`22,028` of those are acyclic and are **coded, not stored** -- their covariant
spec is the source/sink order with each orbit's identical nodes consecutive, a
theorem -- leaving `2,036` stored entries.

## How it is checked

    PYTHONPATH=. python a probe in the source repository \
        --dir dictionaries/flavoured_su4su2 --cutoff 4

Measured 2026-09-16, at `--cutoff 3`: **180/180 stored entries at depth 4, 0
failures**; and on the cluster 2026-09-22: **1,856/1,856 weight-11 stored
entries at depth 5 in 47 s, 0 failures** -- each spec rebuilt against `S` on a
cone strictly wider than its search's, with the `[A_1, D_3]` anchor first as an
abort-on-fail positive control.  The census also checks out cell by cell
against `manifest.json`.

**Tracked in git**, like every permanent tier, at 71 kB gzipped -- see the design notes
"Workflow".
