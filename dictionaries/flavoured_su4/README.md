# The flavoured BPS-quiver dictionary — quivers with a manifest `SU(4)`

**Shard format 3 since 2026-09-28**.  This tier stores ONLY
the **strongly connected** quivers carrying one node orbit of size 4 (a manifest `SU(4)`),
through total unfolded arrow weight 11: **10 strongly connected flavoured quivers in 3.5 kB**: 5 with a covariant spec, 5 `crystalline_only@3`.

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
family through weight 11, 143,598 quivers (20,847 stored), 662 kB.
- It was checked entry by entry against the composition
  (a probe in the source repository): every stored entry is
  answered, no spec is lost, and `S` is reproduced on every sampled pair.
- Its census is kept in the manifest (`census_of_every_flavoured_quiver`).
- The sections below describe that retired tier, and git history keeps its
  shards.

## The retired format-2 tier (the record)

The `SU(4)` member of the flavoured family. **Read
`dictionaries/flavoured/README.md` first** — the construction, the shard format,
the acyclic coding and the checks are identical across the family and only the
flavour group differs.

## What an entry is

A connected BPS quiver carrying **four mutually identical nodes**: nodes that pairwise do not pair
(`B[i][j] = 0`) and every permutation of which is an automorphism of the
exchange matrix, so they are interchangeable and the theory has a manifest
`SU(4)` rotating them.

The selector is the node **orbit** structure, `node_orbits(B, size)` — never a
doublet count.  Three identical nodes are one orbit of size 3 and contain three
*overlapping* doublet pairs; three *disjoint* pairs is `SU(2)³`; four identical
nodes are one orbit of size 4 and not two pairs.  These are different objects and
the selector distinguishes them.

## What is shipped

`--orbit-size 4`, one family.  **Complete through total arrow weight `e ≤ 11`**, the
weight being the **unfolded** quiver's: **143,598 quivers, 662 KB gzipped over
30 shards**.

| certification | entries |
|---|---:|
| `covariant_spec:blocks` | 20,800 |
| `covariant_spec:strip (coded, not stored)` | 122,751 |
| `crystalline_only@3` (no covariant spec found; `Ω(γ, r)` to depth 3) | 47 |
| `crystalline_failed@D` (`Ω` not a character — a conjecture counterexample) | **0** |

`122,751` of those are acyclic and are **coded, not stored** — their
covariant spec is the source/sink order with the identical nodes consecutive, a
theorem — leaving `20,847` stored entries, `47` of which carry `Ω` and no spec.

## How it is checked

    PYTHONPATH=. python a probe in the source repository \
        --dir dictionaries/flavoured_su4 --cutoff 4

Measured on the cluster 2026-09-22: **18,574 / 18,574 weight-11 stored entries at
depth 5, 0 failures** (374 s); the 2026-09-16 record stands for the earlier
weights, **243 / 243 stored entries of weight `e ≤ 9` at depth 4** (`--cutoff 3`).
Each spec is rebuilt against `S` on a cone strictly wider than its search's, with
the `[A_1, D_3]` anchor first as an abort-on-fail positive control.  The census
also checks out cell by cell against `manifest.json`.

## Building more

    PYTHONPATH=. python dictionaries/build_flavoured_by_rank.py \
        --orbit-size 4 --max-weight 12 --workers N --ship <staging dir>

Cells already present are skipped, so this resumes.  Build to a **staging
directory**, never into the tracked tier — that is how the weight-11 cells were
built, one job on Perimeter's `Symmetry` cluster, for which the builder carries
`--shard i/K` / `--merge-shards`; every option but `--no-order-search` leaves a
cell's JSON byte for byte what the plain builder writes.  Operational detail:
the design record.

**Tracked in git**, at 662 KB gzipped — see the design notes "Workflow".
