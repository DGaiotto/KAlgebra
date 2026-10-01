# The flavoured BPS-quiver dictionary — quivers with a manifest `SU(6)`

**Shard format 3 since 2026-09-28**.  This tier stores ONLY
the **strongly connected** quivers carrying one node orbit of size 6 (a manifest `SU(6)`),
through total unfolded arrow weight 12: **no entry at all**: no strongly connected quiver of this weight carries such a family, so every one of its quivers is answered by composition. It climbed from weight 11 to 12 with the redesign: the enumerated tier holds every component it needs.

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
family through weight 11, 3,064 quivers (242 stored), 15 kB.
- It was checked entry by entry against the composition
  (a probe in the source repository): every stored entry is
  answered, no spec is lost, and `S` is reproduced on every sampled pair.
- Its census is kept in the manifest (`census_of_every_flavoured_quiver`).
- The sections below describe that retired tier, and git history keeps its
  shards.

## The retired format-2 tier (the record)

The `SU(6)` member of the flavoured family. **Read
`dictionaries/flavoured/README.md` first** — the construction, the shard format,
the acyclic coding and the checks are identical across the family and only the
flavour group differs.

## What an entry is

A connected BPS quiver carrying a **node orbit of size 6**: six nodes that
pairwise do not pair (`B[i][j] = 0`) and every permutation of which is an
automorphism of the exchange matrix — six *identical, interchangeable* nodes, so
the theory has a manifest `SU(6)` rotating them.

## Why this tier is thin, and what unblocked weight 11

Unfolding an orbit of size `N` at a node of degree `d` adds `(N−1)·d` to the
unfolded weight, so at `N = 6` a single degree-1 node already costs 5.  The
lightest member is the **six-leaf star** at weight 6 (measured: the lowest cell
present is `(n, e) = (7, 6)`), which left only three units of room below weight
9 — 88 quivers, two of them stored.  Weight 10 brings that to 500 quivers with
24 stored; weight 11 to **3,064 with 242 stored**.

**Weight 11 used to be where `canonical_form` stopped**: a build to weight 11
completed ranks 7–11 and then raised `RuntimeError: canonical_form: leaf cap
exceeded` in rank 12, the leaf count running with `|Aut|` — a star on `n` nodes
has `(n−1)!` leaves.  The fold walk now tests the **orbit count before**
canonicalising (`654c8a7d`), and the quivers with the largest `|Aut|` are
exactly the ones that test discards: a 12-node star has `C(11, 6)` orbits of
size 6, never exactly one, so no single-orbit tier admits it.  Rank 12 then
completes, as the cell `(12, 11)` — 702 quivers, all acyclic and so coded rather
than stored.  The yielded walk is unchanged: a test runs the old order beside
the new one and requires identical records.

That is **not** the automorphism pruning `quiver_enumeration`'s docstring and
the 2026-09-13 note call for — still the fix for a quiver this tier *does*
admit, six interchangeable nodes contributing `6! = 720` to `|Aut|`; the
reordering only stops the tier paying for the quivers it rejects.

## What is shipped

`--orbit-size 6`, one orbit.  **Complete through total arrow weight
`e <= 11`**, the weight being the **unfolded** quiver's: **3,064 quivers,
15 kB gzipped over 16 shards** (weight 11 built in shards on Perimeter
Institute's Symmetry cluster, 24 workers).

| certification | entries |
|---|---:|
| `covariant_spec:blocks` | 242 |
| `covariant_spec:strip (coded, not stored)` | 2,822 |

`2,822` of those are acyclic and are **coded, not stored** -- their covariant
spec is the source/sink order with each orbit's identical nodes consecutive, a
theorem -- leaving `242` stored entries.

## How it is checked

    PYTHONPATH=. python a probe in the source repository \
        --dir dictionaries/flavoured_su6 --cutoff 4

Measured 2026-09-16 on what shipped then: **242/242 stored entries at depth 4**
(`--cutoff 3`), 0 failures.  Re-checked on the cluster 2026-09-22 at depth 5:
**218/218 weight-11 stored entries in 8 s, 0 failures**.  Each spec is rebuilt
against `S` on a cone strictly wider than its search's, with the `[A_1, D_3]`
anchor first as an abort-on-fail positive control.  The census also checks out
cell by cell against `manifest.json`.

## Building more

    PYTHONPATH=. python dictionaries/build_flavoured_by_rank.py \
        --orbit-size 6 --max-weight 12 --workers N --ship <staging dir>

Cells already present are skipped, so this resumes; build to a **staging
directory**, never into the tracked tier.

**Tracked in git**, like every permanent tier, at 15 kB gzipped -- see the design notes
"Workflow".
