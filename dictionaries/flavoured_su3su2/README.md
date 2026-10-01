# The flavoured BPS-quiver dictionary — quivers with a manifest `SU(3) x SU(2)`

**Shard format 3 since 2026-09-28**.  This tier stores ONLY
the **strongly connected** quivers carrying a family of disjoint node orbits of sizes 3, 2 (a manifest `SU(3)×SU(2)`),
through total unfolded arrow weight 11: **8 strongly connected flavoured quivers in 2.5 kB**: 8 with a covariant spec, 0 `crystalline_only@3`.

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
family through weight 11, 186,921 quivers (21,711 stored), 684 kB.
- It was checked entry by entry against the composition
  (a probe in the source repository): every stored entry is
  answered, no spec is lost, and `S` is reproduced on every sampled pair.
- Its census is kept in the manifest (`census_of_every_flavoured_quiver`).
- The sections below describe that retired tier, and git history keeps its
  shards.

## The retired format-2 tier (the record)

The `SU(3) x SU(2)` member of the flavoured family. **Read
`dictionaries/flavoured/README.md` first** — the construction, the shard format,
the acyclic coding and the checks are identical across the family and only the
flavour group differs.

## What an entry is

A connected BPS quiver carrying **a triple and a disjoint pair of identical nodes**: nodes that pairwise do not pair
(`B[i][j] = 0`) and every permutation of which is an automorphism of the
exchange matrix, so they are interchangeable and the theory has a manifest
`SU(3) x SU(2)` rotating them.

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

`--orbit-sizes 3,2`, one family.  **Complete through total arrow weight
`e <= 11`**, the weight being the **unfolded** quiver's: **186,921 quivers,
684 kB gzipped over 28 shards** (weight `<= 10` built in 7.8 h, 2 workers,
2.20 GB peak; weight 11 one job on Perimeter's Symmetry cluster, 24 workers).

| certification | entries |
|---|---:|
| `covariant_spec:blocks` | 21,695 |
| `covariant_spec:strip (coded, not stored)` | 165,210 |
| `crystalline_only@3` (no covariant spec found; `Ω(γ, r)` to depth 3) | 16 |

`165,210` of those are acyclic and are **coded, not stored** -- their covariant
spec is the source/sink order with each orbit's identical nodes consecutive, a
theorem -- leaving `21,711` stored entries, sixteen of them crystalline-only,
all in cell `(8, 11)`, and none `crystalline_failed`.

## How it is checked

    PYTHONPATH=. python a probe in the source repository \
        --dir dictionaries/flavoured_su3su2 --cutoff 4

Measured 2026-09-17 for weight `<= 10`: **2,067/2,067 stored entries, no failures, 202 s**, at depth 4 (`--cutoff 3`).
Measured on the cluster 2026-09-22 for weight 11: **19,644/19,644 stored entries, no failures, 439 s**, at depth 5 -- each spec rebuilt
against `S` on a cone strictly wider than its search's, with the `[A_1, D_3]`
anchor first as an abort-on-fail positive control.  The census also checks out
cell by cell against `manifest.json`.

**Tracked in git**, like every permanent tier, at 684 kB gzipped -- see the design notes
"Workflow".
