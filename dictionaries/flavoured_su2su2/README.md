# The flavoured BPS-quiver dictionary — quivers with a manifest `SU(2) x SU(2)`

**Shard format 3 since 2026-09-28**.  This tier stores ONLY
the **strongly connected** quivers carrying a family of disjoint node orbits of sizes 2, 2 (a manifest `SU(2)×SU(2)`),
through total unfolded arrow weight 10: **28 strongly connected flavoured quivers in 5.3 kB**: 27 with a covariant spec, 1 `crystalline_only@3`.

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
family through weight 10, 110,437 quivers (12,738 stored), 424 kB.
- It was checked entry by entry against the composition
  (a probe in the source repository): every stored entry is
  answered, no spec is lost, and `S` is reproduced on every sampled pair.
- Its census is kept in the manifest (`census_of_every_flavoured_quiver`).
- The sections below describe that retired tier, and git history keeps its
  shards.

## The retired format-2 tier (the record)

The `SU(2) x SU(2)` member of the flavoured family. **Read
`dictionaries/flavoured/README.md` first** — the construction, the shard format,
the acyclic coding and the checks are identical across the family and only the
flavour group differs.

## What an entry is

A connected BPS quiver carrying **two DISJOINT pairs of identical nodes**: nodes that pairwise do not pair
(`B[i][j] = 0`) and every permutation of which is an automorphism of the
exchange matrix, so they are interchangeable and the theory has a manifest
`SU(2) x SU(2)` rotating them.

The selector is the node **orbit** structure, `node_orbits(B, size)` — never a
doublet count.  Three identical nodes are one orbit of size 3 and contain three
*overlapping* doublet pairs; three *disjoint* pairs is `SU(2)³`; four identical
nodes are one orbit of size 4 and not two pairs.  These are different objects and
the selector distinguishes them.

**Two disjoint orbits, which is a different object from one big
orbit.** Four mutually identical nodes admit three ways of pairing
into two pairs, so they give three families and are rejected here
rather than filed as this tier's shape; the acceptance is that
`node_orbit_families(B, sizes)` has exactly one element.  Folding
several disjoint orbits at once is `fold_orbits`, which is what the
single-orbit builder could not do while it folded only the first.


## What is shipped

`--orbit-sizes 2,2`, one family.  **Complete through total arrow weight `e ≤ 10`**, the
weight being the **unfolded** quiver's: **110,437 quivers, 424 KB gzipped over
27 shards**.

| certification | entries |
|---|---:|
| `covariant_spec:blocks` | 12,726 |
| `covariant_spec:strip (coded, not stored)` | 97,699 |
| `crystalline_only@3` | 12 |
| `crystalline_failed@D` (`Ω` not a character — a conjecture counterexample) | **0** |

`97,699` of those are acyclic and are **coded, not stored** — their
covariant spec is the source/sink order with the identical nodes consecutive, a
theorem — leaving `12,738` stored entries, twelve of which carry `Ω(γ, r)` in
place of a spec.

## How it is checked

    PYTHONPATH=. python a probe in the source repository \
        --dir dictionaries/flavoured_su2su2 --cutoff 4

Measured on the `Symmetry` cluster 2026-09-22: **11,516 / 11,516 weight-10
stored entries at depth 5, 0 failures** (172 s); the earlier weights 2026-09-16,
**1,222 / 1,222 stored entries at depth 4, 0 failures**.  Each spec is rebuilt
against `S` on a cone strictly wider than its search's, with the `[A_1, D_3]`
anchor first as an abort-on-fail positive control.  The census also checks out
cell by cell against `manifest.json`.

## Building more

    PYTHONPATH=. python dictionaries/build_flavoured_by_rank.py \
        --orbit-sizes 2,2 --max-weight 11 --workers N --ship <staging dir>

Cells already present are skipped, so this resumes.  Build to a **staging
directory**, never into the tracked tier.  Operational detail:
the design record.

**Tracked in git**, at 424 KB gzipped — see the design notes "Workflow".
