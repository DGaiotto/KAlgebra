# The flavoured BPS-quiver dictionary — quivers with a manifest `SU(5)`

**Shard format 3 since 2026-09-28**.  This tier stores ONLY
the **strongly connected** quivers carrying one node orbit of size 5 (a manifest `SU(5)`),
through total unfolded arrow weight 11: **1 strongly connected flavoured quivers in 0.6 kB**: 1 with a covariant spec, 0 `crystalline_only@3`.

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
family through weight 11, 20,373 quivers (2,267 stored), 79 kB.
- It was checked entry by entry against the composition
  (a probe in the source repository): every stored entry is
  answered, no spec is lost, and `S` is reproduced on every sampled pair.
- Its census is kept in the manifest (`census_of_every_flavoured_quiver`).
- The sections below describe that retired tier, and git history keeps its
  shards.

## The retired format-2 tier (the record)

The `SU(5)` member of the flavoured family. **Read
`dictionaries/flavoured/README.md` first** — the construction, the shard format,
the acyclic coding and the checks are identical across the family and only the
flavour group differs.

## What an entry is

A connected BPS quiver carrying **five mutually identical nodes**: nodes that pairwise do not pair
(`B[i][j] = 0`) and every permutation of which is an automorphism of the
exchange matrix, so they are interchangeable and the theory has a manifest
`SU(5)` rotating them.

The selector is the node **orbit** structure, `node_orbits(B, size)` — never a
doublet count.  Three identical nodes are one orbit of size 3 and contain three
*overlapping* doublet pairs; three *disjoint* pairs is `SU(2)³`; four identical
nodes are one orbit of size 4 and not two pairs.  These are different objects and
the selector distinguishes them.

## What is shipped

`--orbit-size 5`, one family.  **Complete through total arrow weight `e ≤ 11`**, the
weight being the **unfolded** quiver's: **20,373 quivers, 79 KB gzipped over
23 shards**.

| certification | entries |
|---|---:|
| `covariant_spec:blocks` | 2,265 |
| `covariant_spec:strip (coded, not stored)` | 18,106 |
| `crystalline_only@3` (no covariant spec found within budget) | 2 |
| `crystalline_failed@D` (`Ω` not a character — a conjecture counterexample) | **0** |

`18,106` of those are acyclic and are **coded, not stored** — their
covariant spec is the source/sink order with the identical nodes consecutive, a
theorem — leaving `2,267` stored entries.

## How it is checked

    PYTHONPATH=. python a probe in the source repository \
        --dir dictionaries/flavoured_su5 --cutoff 4

Measured 2026-09-16, when the tier stopped at weight 9: **24 / 24 stored entries
at depth 4, 0 failures**; on Symmetry 2026-09-22, the weight-11 entries,
**2,025 / 2,025 at depth 5, 0 failures** (45 s) — each spec rebuilt against `S`
on a cone strictly wider than its search's, with the `[A_1, D_3]` anchor first
as an abort-on-fail positive control.  The census also checks out cell by cell
against `manifest.json`.

## Building more

    PYTHONPATH=. python dictionaries/build_flavoured_by_rank.py \
        --orbit-size 5 --max-weight 12 --workers N --ship <staging dir>

Cells already present are skipped, so this resumes.  Build to a **staging
directory**, never into the tracked tier.  Weight 11 reaches rank 12, which
clears only because the fold walk tests the orbit count before `canonical_form`
(654c8a7d).  Operational detail:
the design record.

**Tracked in git**, at 79 KB gzipped — see the design notes "Workflow".
