# The flavoured BPS-quiver dictionary — quivers with a manifest `SU(2)`

**Shard format 3 since 2026-09-28**.  This tier stores ONLY
the **strongly connected** quivers carrying one node orbit of size 2 (a manifest `SU(2)`),
through total unfolded arrow weight 10: **1,154 strongly connected flavoured quivers in 31.2 kB**: 1,092 with a covariant spec, 62 `crystalline_only@3`.

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
family through weight 10, 1,022,925 quivers (200,173 stored), 6.32 MB.
- It was checked entry by entry against the composition
  (a probe in the source repository): every stored entry is
  answered, no spec is lost, and `S` is reproduced on every sampled pair.
- Its census is kept in the manifest (`census_of_every_flavoured_quiver`).
- The sections below describe that retired tier, and git history keeps its
  shards.

## The retired format-2 tier (the record)

**What this is.** A **refinement** of the enumerated dictionary
(`dictionaries/enumerated/`, the design record), not a second enumeration. An entry is a
connected BPS quiver that carries a **node doublet** — a transposition of two
nodes that is an automorphism of the exchange matrix, the two nodes not pairing
with each other — so those two nodes are identical and permutable and the theory
has a manifest `SU(2)` rotating them (the author's *"as soon as you have `N`
identical nodes in the quiver, the theory has manifest `SU(N)` symmetry"*,
2026-09-13). Each entry carries, in the author's order of preference:

1. an **`SU(2)`-covariant spec** — a flavoured spec in the sense of
   `src/bps/flavoured_spec.py`, whose factors for one irrep are consecutive, so the
   whole factorisation is a product of
   `E^{(s;r)}_𝖖(X_γ) = ∏_{w∈r} E^{(s)}_𝖖(μ^w X_γ)` and the flavour symmetry is
   manifest factor by factor;
2. otherwise the **`SU(2)`-covariant crystalline `S`** to a stated depth —
   `Ω(γ, r)`, a palindromic `𝖖`-polynomial per irrep at each reduced charge.

Keying on the **unfolded** quiver (the honest BPS quiver of the theory) rather
than on the folded one is what makes the refinement statement checkable: cells
line up `(n, e)` with the unflavoured tier and every entry is by construction
already in it, a set built by a different route.

The axiomatics behind the entries is `notes/flavoured_s_axioms.md` (the `\cE_G`
construction); the builder is `dictionaries/build_flavoured.py`; the reader is
`dictionary_loader` (`default_flavoured_dictionary_dir`,
`iter_flavoured_entries`, `lookup_flavoured`).

## What is shipped

`doublets = 1` (the `SU(2)` set). **Complete through weight 10** — every connected
quiver with total arrow weight `e = Σ_{i<j} |b_ij| ≤ 10` carrying exactly one node
doublet, up to node permutation.  Weight 10 was finished on Perimeter's Symmetry
cluster on 2026-09-22 (ranks 7–11; the record is the design notes).

**The weight is the UNFOLDED quiver's**, and so is the rank: a shard
`f_n{NN}_e{EE}` is keyed by the honest BPS quiver, not by its fold. Unfolding
duplicates the doublet node, so `e_unfolded = e_folded + d` with `d` that node's
degree — the `[A_1, D_3]` anchor is `(n, e) = (3, 2)` and its fold is `(2, 1)`.

1,022,925 quivers, **6,322,386 B gzipped** (6.32 MB) over 41 shards — 723,200 B
for weight `≤ 9` and 5,599,186 B for weight 10.  (Quote that sum, not `du -h`,
whose 4 KiB block rounding over many small files inflates it.)

| certification | entries |
|---|---:|
| `covariant_spec:strip` (acyclic; **coded**, not stored) | 822,752 |
| `covariant_spec:blocks` (block-mutation bidirectional BFS) | 196,858 |
| `covariant_spec:order@5` (order search, confirmed to degree 5) | 2,649 |
| `crystalline_only@3` (no covariant spec found; `Ω(γ, r)` to depth 3) | 666 |
| `crystalline_failed@D` (`Ω` not a character — a conjecture counterexample) | **0** |

So 1,022,259 of 1,022,925 have a covariant spec, and **no entry anywhere in the
set fails to have `Ω` a `G`-character** — which is the substantive flavoured claim
(`notes/flavoured_s_axioms.md` §3, where it is an *output* of the recursion and never
imposed) holding on 1,022,925 quivers.

**`manifest.json` is where to read what is claimed.** Its `completeness` now
states `e ≤ 10` with no `coverage` record, because no cell of that weight is
missing.  Continuing the tier is
the design record.

Two things about weight 10 are worth knowing.  The **refinement statement stops**
there, because `dictionaries/enumerated/` ships to `E = 9` and the flavoured tier
now reaches past it, so a weight-10 entry has no unflavoured cell to be checked
against.  And the **crystalline entries concentrate at rank 7**: 613 of weight
10's 177,791 stored entries (0.34 %, against 0.24 % at weight 9), of which 378
are in cell (7,10) and none at all in ranks 8–11.  An earlier reading of the
first four cells put that rate at 4.6 %; those were ranks 3–6, the dense low
ranks, and the whole weight does not behave that way.

**What weight 10 cost, and where.** 30 core-hours for rank 7 against 7.5 for
rank 8, although rank 8 has more than twice the stored entries: 99.6 % of rank 7
went into the block BFS, and 93 % into the 378 entries it *fails* on — about
266 s each, up to 465 s — after which the order search adds 0.37 % and finds
nothing.  Rank 7 is also where the builder ran out of memory with 78 workers on
a 180 GB node, because those same entries are the memory-hungry ones; it
finished on 8 workers.  Ranks 8–11 are 0.4 s an entry with no crystalline
entries at all.

## Shard format 2 — the acyclic quivers are coded, not stored

Same discipline as the unflavoured tier. On an **acyclic** flavoured quiver the
source/sink order with the two doublet nodes consecutive is a covariant spec —
a theorem, not a search: the two nodes are identical and non-adjacent, so they
have the same relations to every other node and can always be moved next to each
other without disturbing the order, and consecutive is exactly what lets the two
factors `E_𝖖(μ^{+}X_γ)` and `E_𝖖(μ^{-}X_γ)` group into one flavoured generator.
At the reduced level it is simply the folded quiver's source/sink order, one
generator per reduced node — the floor.

So those entries are **recomputed on lookup** rather than written (80 % of the
set), and each cell header keeps `count = acyclic + stored`, leaving the
completeness statement unchanged and checkable. **A scan that forgets
`include_acyclic=True` silently omits them** — and with them the `[A_1, D_3]`
anchor, the canonical one-`SU(2)` quiver.

Shard shape (gzipped JSON, self-describing so a single cell is usable alone):

```
{"format": 2, "n": 5, "e": 5, "count": 36, "acyclic": 34, "stored": 2,
 "acyclic_coded": true,
 "entries": [{"exchange": [[…]],        # the UNFOLDED quiver, canonical form
              "doublet": [0, 1],        # the node pair, in canonical order
              "reduced": [[…]],         # the folded (flavoured) quiver
              "factors": [2, 1, 1, 1],  # per reduced node: 1, or 2 at the doublet
              "certification": "covariant_spec:order@5",
              "spec": [[charge, irrep], …]}]}   # or "omega", never both
```

`manifest.json` carries `doublets`, `max_weight`, the per-cell census, the
totals and the certification census.

## How it is checked

* **In CI** — the suite in the source repository: the census adds up cell by cell
  and in the aggregate, every entry carries a recognised certification, a
  deterministic sample of the committed specs is rebuilt against `S`, and the
  loader answers in any node order.
* **Exhaustively** — a probe in the source repository
  re-checks **every** stored entry.  Measured 2026-09-15/16: **22,382 / 22,382 at
  weight 9** and **5,140 / 5,140 on the first four weight-10 cells**, depth 4,
  0 failures; and on Symmetry 2026-09-22, the rest of weight 10 —
  **172,651 / 172,651** stored entries at depth 5 (`--cutoff 4`), 0 failures
  (the crystalline-only entries have no spec to rebuild and are reported at
  depth 0).  It checks: the fold is consistent with the recorded
  unfolded quiver, the entry is in the unflavoured dictionary, and the spec
  rebuilds `S` on a cone **strictly wider** than the search's (an in-cone
  rebuild is a screen, not a certificate — the audit, where three
  fixed depths were refuted). The `[A_1, D_3]` anchor runs first as a positive
  control and the run aborts if it fails.

**Tracked in git**, like the enumerated tier, at 6.32 MB gzipped — see the design notes
"Workflow" for the standing ruling on `dictionaries/` artifacts and the
per-dictionary budget it records.
