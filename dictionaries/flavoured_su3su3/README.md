# The flavoured BPS-quiver dictionary — quivers with a manifest `SU(3) x SU(3)`

**Shard format 3 since 2026-09-28**.  This tier stores ONLY
the **strongly connected** quivers carrying a family of disjoint node orbits of sizes 3, 3 (a manifest `SU(3)×SU(3)`),
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
family through weight 11, 11,835 quivers (907 stored), 39 kB.
- It was checked entry by entry against the composition
  (a probe in the source repository): every stored entry is
  answered, no spec is lost, and `S` is reproduced on every sampled pair.
- Its census is kept in the manifest (`census_of_every_flavoured_quiver`).
- The sections below describe that retired tier, and git history keeps its
  shards.

## The retired format-2 tier (the record)

The `SU(3) x SU(3)` member of the flavoured family. **Read
`dictionaries/flavoured/README.md` first** — the construction, the shard format,
the acyclic coding and the checks are identical across the family and only the
flavour group differs.

## What an entry is

A connected BPS quiver carrying **two disjoint triples of identical nodes**: two
orbits of size 3, each a set of nodes that pairwise do not pair (`B[i][j] = 0`)
and every permutation of which is an automorphism, and the two sharing no node.

It is the first member with **two non-abelian factors of equal rank above one**,
so it is the sharpest test in the family of the emergent claim of
`notes/flavoured_s_axioms.md` section 3 — `Omega` takes values in
`R(SU(3)) x R(SU(3))` tensor `P`, and the multiplicities have to assemble into
honest characters of *each* factor separately.  At weight 9 the tier held 230
quivers with 5 stored, which tested very little; weight 10 brought it to 1,611
with 78 stored (2 h 45 m, 1.19 GB peak), weight 11 to 11,835 with 907 stored.

## What is shipped

`--orbit-sizes 3,3`, one family.  **Complete through total arrow weight
`e <= 11`**, the weight being the **unfolded** quiver's: **11,835 quivers,
39 kB gzipped over 18 shards**.  Weight 11 was finished on Perimeter's Symmetry
cluster on 2026-09-22.

| certification | entries |
|---|---:|
| `covariant_spec:blocks` | 907 |
| `covariant_spec:strip (coded, not stored)` | 10,928 |

`10,928` of those are acyclic and are **coded, not stored** -- their covariant
spec is the source/sink order with each orbit's identical nodes consecutive, a
theorem -- leaving `907` stored entries.

## How it is checked

    PYTHONPATH=. python a probe in the source repository \
        --dir dictionaries/flavoured_su3su3 --cutoff 4

Measured 2026-09-16: **78/78 stored entries at depth 4, 0 failures** on the
weights through 10 (`--cutoff 3`).  On the cluster, 2026-09-22: **829/829 of
the weight-11 stored entries at depth 5 in 24 s, 0 failures**.  Each spec is
rebuilt against `S` on a cone strictly wider than its search's, with the
`[A_1, D_3]` anchor first as an abort-on-fail positive control.  The census
also checks out cell by cell against `manifest.json`.

**Tracked in git**, like every permanent tier, at 39 kB gzipped -- see the design notes
"Workflow".
