# The enumerated BPS-quiver dictionary — permanent tier

**What this is (shard format 3, since 2026-09-24).**
Exhaustive shards of `(strongly connected BPS quiver, one spec)`: for every
rank `n` and every total arrow weight `e ≤ E`, the file `q_n{NN}_e{EE}.json.gz`
holds **every strongly connected BPS quiver with `n` nodes and `e` arrows
counted with multiplicity** (`e = Σ_{i<j} |b_ij|`), up to node permutation.
Each has either one accepted spec (a finite ordered list of charges with
`S = ∏_a E_𝖖(X_{γ_a})`, the shortest known) or, where no producer reached
one, the crystalline certification.  **`E = 12`: 728,921 quivers, 674,597
with a green spec and 54,324 `crystalline_only@6`, in 6.3 MB** — against 8.9 MB for the weight-10 tier of every connected quiver it
replaces, which went only through weight 10.

| weight | quivers | green spec | `crystalline_only@6` |
|---|---:|---:|---:|
| ≤ 10 | 18,289 | 17,308 | 981 |
| 11 | 93,698 | 87,661 | 6,037 |
| 12 | 616,934 | 569,628 | 47,306 |

A quiver is *strongly connected* when every node reaches every other along
arrows.  Equivalently (the author's phrasing, 2026-09-23), it cannot be split
into two nonempty parts with every arrow between them going the same way.

**Coverage: every other quiver is answered by composition.**
`dictionary_loader.lookup_enumerated(B)` covers every BPS quiver — connected
or not, of any rank and any total weight — whose strongly connected
components have total arrow weight `≤ E`.
- It splits `B` into its strongly connected components.
- It looks each one of two or more nodes up here.
- It returns the components' specs concatenated in a source-first order, a
  single node contributing its own charge (`spec_acceptance.spec_from_components`).
- Green component specs compose to a green spec (a theorem,
  `notes/strongly_connected_redesign.md`
  §1.1), and so do the axioms to a given depth (§1.2).
- The answer's record is composed over the components of two or more
  nodes: `green` iff every one is; `axioms@D` at the least depth; the least
  `crystalline_only@D` if one is; a component's `axiom_mismatch@D` carried.
- An acyclic quiver — every component a single node — is answered `green`
  with its source/sink order.

This covers the whole of the retired tier.  It was checked entry by entry
(a probe in the source repository; the record is
a probe in the source repository):
- all 1,253,781 of its stored quivers are answered, green wherever they were
  green, with 207 of its spec-less ones upgraded to green;
- `S` agreed on every sampled pair;
- its 3,680,590 coded acyclic quivers are answered by exactly the
  source/sink order it coded — checked on all 12,465 of weight ≤ 7, and by
  construction beyond.

**What a record claims.**
- **`green;axioms@D`:** the spec replays as a negating sequence (exact), and
  the `S` axioms were checked on the cone to degree `D`.  **Every shipped
  spec carries one; no bare `green` ships**.
  - `green;axioms@5` on the 674,012 of weights 9–12 (strips at 3).  The deferral of the `S`-axiom check was
    lifted for the weekend of 2026-09-26.  Weight 12's records are the
    Symmetry cluster's run where the spec is the same one, and the
    iMac's pass on the rest.
  - `green;axioms@7` on the 585 of weight ≤ 8, T26's records, carried.
  - 0 mismatches.  A green spec whose check failed would be a conjecture
    counterexample, listed under the manifest's `axiom_mismatches`.
- **`crystalline_only@D`:** NO accepted spec was found, and the crystalline
  `S` was computed to depth `D`.  **It is not a proof that no spec exists.**
  - The search is recorded in the manifest (`search`): seeds, propagation
    (tail cap 5,000), and BFS.  Every spec-less entry was searched at least
    to depth 12 at 60 s.  6,600 were searched deeper, by `find_spec_auto` to
    depth 30 at 120–1,200 s: 700 exhausted and 5,900 timed out.  The order
    search did not run (it is skipped from weight 10 on).
  - A depth-12 BFS rules out only sequences of length ≤ 12, at any rank.  The
    "2n − 2" pattern holds for the oriented n-cycle, not in general.
  - **Low-rank long sequences exist, and they are reached by propagation from
    higher weight, not by search**.  On 2026-09-26, propagation from
    weight-13 specs gave 156 specs at rank ≤ 7 on weight-12 quivers where a
    depth-12 BFS had exhausted.  154 of them are longer than 12 mutations, up to
    21 at rank 7.  Search found none of these: `find_spec_auto` to depth 30 found
    0 of 976 such quivers at weight ≤ 10 and 0 of 153 sampled at weight 11.
  - The manifest's `search.spec_less_entries_by_bfs_attempt` says how deeply,
    and with which strategy, each spec-less entry was searched.

**The census of every connected quiver** (`count`, `acyclic`, `stored` per
cell) is kept in the manifest for `e ≤ 10`, as the record of the retired tier
and the cross-check of the composition (`census_of_every_connected_quiver`).
Above weight 10 it is not enumerable on the author's machine: there are about
36 M connected quivers at weight 11.

**Build and ship.**  The tier is shipped from a strongly connected rich
build by `dictionaries/build_enumerated.py::ship_strongly_connected` (format
3, only through a weight whose every entry is settled).  The build is
`--strongly-connected`: the walk over strongly connected quivers, seeded from
an earlier tier (`--seeds-tier`).  It climbs in weight:
- **weight 12:** `SC12`, shipped here 2026-09-27, and
  checked entry by entry against the weight-11 tier.  No spec was lost and
  none got longer: an earlier tier's spec that is as deeply certified and
  strictly shorter is kept;
- **weight 13:** 4,274,199 strongly connected quivers (the census of
  2026-09-24), about 42 MB through weight 13.  Its specs come by
  propagation, on Symmetry.

**Iteration.**  `iter_enumerated_entries()` yields the stored strongly
connected entries.  Iterating every connected quiver means enumerating them
and looking each one up, which the loader leaves to the caller
(`include_acyclic=True` explains this for a format-3 tier).

## History

The sections below describe the tier this one replaced: every connected
quiver through weight 10, shard format 2.  Git history keeps its shards (the
last commit holding them precedes shard format 3).

### Shard format 2 (2026-09-13) — compact entries, acyclic quivers coded

Two changes took the weight-8 tier from 1.15 MB to 248 KB (4.6×), measured in
a probe in the source repository and
the design record §12:

1. **A compact entry.**  The quiver is its **arrow string** — one character
   pair per arrow, `2e` characters, node index in base 36, direction from the
   sign of `b_ij` — and a green spec is the **node-index string** of its
   replay, one character per factor.  The exchange matrix and the charges are
   rebuilt by the loader (`matrix_from_arrows`, `spec_from_sequence`: a few
   dozen mutations, sub-millisecond).  The arrow list is also the right
   logical object, since its length *is* the cell's weight.
2. **The acyclic quivers are coded, not stored**.  For an acyclic quiver the
   source/sink order is a spec — a theorem, `spec_acceptance.strip_spec` — so
   `lookup_enumerated` answers one **without opening a file**, and those
   entries (82 % of the set at weight 8) are not written.  Each cell header
   keeps the census, `count = acyclic + stored`, so the completeness statement
   is unchanged and checkable; `iter_enumerated_entries(include_acyclic=True)`
   regenerates the coded half by re-enumerating.

An acyclic entry is coded only where its stored spec **is** a source/sink
order (`is_source_sink_order`); anything else is stored as usual.  That is
what lets the build's certification transfer to the order a lookup computes:
two source/sink orders differ by transpositions of nodes with no arrow between
them, whose `E_𝖖` factors commute, so they give the same `S` (pinned in
the suite in the source repository).
One entry of the weight-8 set is acyclic with a spec of another shape and is
therefore stored, not coded.

**Shard shape** (gzipped JSON, self-describing so a single cell is usable
alone):

```
{"plan": "41_enumerated_quiver_dictionary", "format": 2, "n": 5, "e": 6,
 "completeness": "every connected BPS quiver with n nodes and e arrows …",
 "acyclic_coded": "the acyclic quivers of this cell are CODED, not stored …",
 "count": 636, "acyclic": 500, "stored": 136, "spec_verified": 636,
 "accept_depth": 5, "verify_depth": 5, "verify_strip_depth": 3,
 "entries": [{"q": "0102122334",      # arrow string: 2e chars, pairs src,dst
              "s": "10213",           # node-index string of the green replay
              "c": "green;axioms@7"}, … ]}
```

`v` replaces `s` on an entry whose spec is not a negating sequence in its
stored order (no index string exists; none has appeared).  Neither key
means no accepted spec, and the certification says `crystalline_only@D`.
A format-1 shard (`{exchange, spec, certification}`, every quiver stored) still
loads unchanged.

At `E = 10` the counts are 1,253,781 stored and 3,680,590 coded.  The stored
entries are bare `green` on 1,101,677 (1,101,675 of weight 10, and 2 of weight
9 that the weight-9 tier had as crystalline-only and this build reached with a
green spec), `green;axioms@5` on 127,931 (all of weight 9), `green;axioms@7` on
17,374 (weight ≤ 8: the T26 records, carried with their specs) and
`crystalline_only@6` on 6,799 (6,148 of weight 10, 651 of weight ≤ 9 — the
weight-9 tier's 653 less those 2).  The coded half's certifications are bare
`green` on 3,157,622 (all of weight 10), `green;axioms@7` on 55,258,
`green;axioms@5` on 443,171 and `green;axioms@3` on 24,539 (a coded entry is
checked at the strip depth, 3, when its primary is the strip order, and at the
full depth, 5, when a propagated source/sink order is its primary).  At
`E = 9` the tier was 145,958 stored and 522,968 coded, 0.99 MB.

`certification` records what held for the shipped spec:

| value | meaning |
|---|---|
| `green` | the spec replays as a genuine negating sequence — exact, finite, combinatorial (`spec_acceptance.is_green_sequence`); the `S` axioms are ASSUMED for it (the build's working conjecture, user 2026-09-13: *"try build assuming every conjecture is true, and then any contradiction encountered when building is a conjecture test"*) |
| `green;axioms@D` | green AND the axiom check passed at cone depth `D`: the BPS quiver constraint `S = 1 − 𝖖 Σ_a X_a + O(𝖖²)` swept over the whole truncated cone plus agreement with the crystalline `S` (`spec_acceptance.accept_spec`) |
| `green;axiom_mismatch@D` | green but the check FAILED at depth `D` — a conjecture counterexample, listed in `manifest.json` under `axiom_mismatches` |
| `axioms@D` | not green; accepted by the axiom check alone, in whatever factor order |
| `crystalline_only@D` | no producer found a finite factorisation; the loader builds the entry spec-free from the crystalline `S` |

The coded acyclic entries carry their certification as a per-cell histogram
(`acyclic_certifications`, in each shard header and summed in the manifest)
rather than per entry.  The weight ≤ 8 cells carry the depth-7 histogram of
the T26 pass except `n = 7, e = 8`: that cell's earlier histogram holds the
three rank-7 strip specs the 120 s budget left at depth 3, and since the coded
entries are not stored individually a histogram is carried whole only when
every depth it records is at least every depth the new build recorded — there
the new build reached 5, so the cell keeps this build's own records
(`green;axioms@5` on 22,332, `green;axioms@3` on 802).

**Read the depth.**  `axioms@D` is a FINITE-DEPTH statement: the
constraint — every coefficient of `S` other than the node ones is `O(𝖖²)` —
was checked on the cone charges of degree `≤ D`, and the crystalline
comparison likewise.  Nothing here certifies it at all degrees; whether a
*finite* test can exist is an open research goal.  Green does not certify it either: that a genuine negating
sequence's product satisfies the constraint is the paper's own
"far from obvious" half (`notes/pronilpotent_group_conjecture.md` §6c) — which is
exactly why the build records both, and why a mismatch is a finding.

**Reading it.**

```python
from dictionary_loader import (iter_enumerated_entries, lookup_enumerated,
                               entry_to_bpskalgebra, load_shard)
for entry in iter_enumerated_entries():          # the STORED entries, cells in (n, e) order
    A = entry_to_bpskalgebra(entry)              # a BPSKAlgebra, spec-free if spec is null
for entry in iter_enumerated_entries(include_acyclic=True):
    ...                                          # every connected quiver of the set
entry, spec = lookup_enumerated([[0, 1], [-1, 0]])   # any node order, acyclic or not;
                                                     # spec comes back in YOURS
shard = load_shard("dictionaries/enumerated/q_n03_e03.json.gz")   # header + expanded entries
raw = load_shard(".../q_n03_e03.json.gz", expand=False)           # the compact entries
```

**Regenerating.** The rich, semi-permanent build (spec families with
producer / provenance / acceptance records, exits, the class table, the
provenance tail) is untracked and lives in `dictionaries/enumerated_build/`:

```
PYTHONPATH=. python dictionaries/build_enumerated.py --E 9 --verify-axioms 5 --verify-strip-depth 3 --workers 3 --no-family-test --enumeration-cache e9_records.json --ship
```

rebuilds the weight ≤ 9 set at the depths shipped here (`green;axioms@5`,
strip specs at 3): ≈ 5.3 h of build stages plus a 16 h verification pass on three workers
of the remote session, inside a 14.3 GB memory cgroup, checkpointed and
resumable with `--resume`.  The `e ≤ 8` depth-7 records come from the
weight-8 pass (`--E 8 --verify-axioms 7 --verify-strip-depth 7 --workers 3`:
2.7 h of build and a 10.8 h pass) and reach this tier through
`--keep-deeper-from` (below).  `--no-family-test` leaves out the family
comparison — every alternative spec of an entry against its primary at the
verify depth, a consistency test rather than a condition of shipping; at
weight 9 it is ≈ 2 M spec products with no progress line and no checkpoint,
and neither weight-9 build ran it.  Without `--verify-axioms` the build is
faster and every green spec ships as bare `green`.  For larger weights on a machine or a
cluster: `--workers N` runs the enumeration walk, the BFS fallback, the order
search, the verification pass and the class index in N forked workers (the
result is the same as the sequential build, records and order included);
`--enumeration-cache FILE` saves the stage-0 records once and reloads them on
every restart or shard (14 min at weight 9 becomes seconds); `--shard i/K`
builds one of K slices of the set: every shard computes the same strip +
propagation closure over all entries and runs the per-entry stages (BFS, order
search, verification, content) on the entries with index ≡ i mod K only, and
`--merge-shards DIR…` joins the K rich builds at `--out` (each entry from its
owner), re-runs the post-BFS and post-order-search propagation rounds on the
whole set, runs the class index once and ships with `--ship`;
`--max-tail-entries N` sets the propagation tail's cap (default 5,000; at
weight 7 a 20× larger tail cut the BFS fallback by a fifth for a ninefold
closure) and `--family-max K` caps the specs stored per entry (default
2n + 1; the alternatives triple the rich build on disk and feed the family
test, so a cap trades that evidence for disk); mid-stage checkpoints rewrite
only the cells touched since the last write; `--resume
--reverify` re-runs the verification pass deeper on a finished build, skipping
entries already checked to the requested depth.  `--ship-from BUILD_DIR --ship`
re-packages an existing rich build — which is how deeper records reach the
shipped set: the depth-7 certifications here come from `--resume --reverify
--verify-axioms 7 --verify-strip-depth 7` on the finished weight-8 build
(55,617 of 55,620 stored green specs re-checked at depth 7, three timeouts, 0
mismatches; the 2,209 family pairs all agree), then `--ship-from`.
`--keep-deeper-from SHIPPED_DIR` keeps, when shipping, an already-shipped
tier's certification wherever it is strictly deeper than this build's: a
stored entry is carried WITH its spec (a differently seeded build usually
reaches another spec for the same quiver, and a record belongs to one spec —
17,374 here, 15,601 of them with the earlier spec), and a cell's coded-half
histogram is carried whole when the cell's membership is unchanged and every
depth it records is at least every depth this build recorded.
`--recompact DIR` re-encodes an already-shipped tier from format 1 to format 2
without a rich build, carrying every certification across verbatim — which is
how this tier first changed format, before the depth-7 pass covered the whole
build (equivalence checked entry by entry in
a probe in the source repository).  The header fields
`accept_depth` (the axiom check for non-green candidates), `verify_depth` and
`verify_strip_depth` (the post-build check on green specs, all others / the
strip specs) say what the build ran; the depth that actually held for an entry
is in its certification.  Design, rulings and measurements:
the design record.

**Weight 10** was built on the author's machine by the overnight driver
`scripts/nightly_dictionary_build.sh` (a LaunchAgent, 19:00–07:00 on weeknights
and continuous at weekends), resuming nightly with

```
PYTHONPATH=. python dictionaries/build_enumerated.py --E 10 --resume --workers 5 --enumeration-cache dictionaries/enumerated_build/E10/enum_E10.json --verify-axioms 0 --checkpoint-every 0 --checkpoint-interval 60 --timeout 30 --family-max 2 --no-order-search --seeds-dir dictionaries_weak --seeds-dir dictionaries/enumerated_build/harvest/final_v1 --out dictionaries/enumerated_build/E10 --stop-at <the window's close, as YYYY-MM-DDTHH:MM>
```

(the last two nights: BFS on the final 25,151 quivers in 9.1 h on five
workers, then 22 min of propagation, 21 min of crystalline content and a
40 min class index — 130,568 mutation classes inside `e ≤ 10`), and shipped
over the weight-9 tier, keeping every deeper record, with a copy of that tier
as the source:

```
cp -R dictionaries/enumerated "$TMPDIR/enumerated_keep_src"
PYTHONPATH=. python dictionaries/build_enumerated.py --ship-from dictionaries/enumerated_build/E10 --ship --keep-deeper-from "$TMPDIR/enumerated_keep_src"
```

(20 min, 14.8 GB peak resident).  The deferred verification is
`--resume --verify-axioms 5 --verify-strip-depth 3` on the rich build, about
150 core-hours, followed by the same ship.
