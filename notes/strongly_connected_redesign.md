# Plan 41 — proposal: the dictionary re-designed around strongly connected quivers

**Status: DESIGN RULED 2026-09-23 (D33–D39 in `decisions.md`); the spec finder
(§5, D37) IMPLEMENTED the same day with T37, and the S-builder's default order
made to follow the decomposition (D44, T38); the tier's migration (§6) is
next.**  The user adopted every item and reversed one recommendation:
the tier climbs in weight (D35, *"Why? we can now climb in weight with the same
footprint."*).  The order of work is the user's: the effect of the spec-finder
change (§5) on `BPSKAlgebra`'s methods comes first (T37), then the migration of
§6.  (D31 is taken by the open PR #1533.)

User request (verbatim): *"Propose a re-design of the dictionary around strongly
connected quivers"*, and while it was drafted: *"It may make the current
dictionary very obsolete"*, *"focus on unflavoured."*, *"true, this hits even
find_negating_sequence!"*, *"who cares if we find slightly bigger specs, they are
more principled and we can always shorten afterwards if we care about that."*,
*"and I bet we will usually find shorter specs"*.  It grew out of the user's
question the same day: if the nodes of `Q` split into parts `A` and `B` with
every arrow between them going from `A` to `B`, does `S(A)·S(B)`, in the
correct order, satisfy the `S(Q)` axioms — and likewise for a decomposition into
parts `A_i` that form an acyclic graph, which should then be handled *"by a
function like acyclic quivers already are"*?

Every number in §§1–6 is reproduced by `experiments/strongly_connected_census.py`
(read-only on `dictionaries/enumerated/`; its output is
`experiments/strongly_connected_census.json`).  The flavoured figures of §8
come from a one-off census that was not kept, the re-design being unflavoured.

**Vocabulary.**  *Strongly connected*: every node reaches every other along
arrows (arrow `i → j` iff `B[i][j] > 0`); a *strongly connected component* is a
maximal such set of nodes.  Standard graph theory in its standard meaning, and
the user's own word for the object.  The components form a graph with an arrow
`C → C'` whenever some arrow of `Q` goes from `C` to `C'`; that graph has no
directed cycle, and a **source-first order** of the components is one in which
every arrow between two components points from the earlier to the later — the
order `strip_spec` uses for nodes.  No other new term is used; the function
names are D38's.  Equivalently (the user's characterisation, 2026-09-23: *"It
should mean it cannot be decomposed into two pieces such that all edges between
the two go in the same direction, right?"* — yes): a quiver is strongly
connected iff its nodes admit no split into two non-empty parts with every
arrow between them pointing the same way (no arrows at all included), and
splitting repeatedly until no part splits ends at the strongly connected
components, whatever the order of the splits (pinned in
`tests/test_quiver_enumeration.py`).

## 0. The proposal in one paragraph

The spec, the spectrum generator and the crystalline content of a BPS quiver are
determined by those of its strongly connected components, taken in source-first
order (§1: for green specs a theorem with no conjecture; for the axioms a
theorem; for the crystalline content a consequence of the factor engine's own
rule).  So the dictionary needs to **store only strongly connected quivers**,
and every other quiver — the acyclic ones included, which are the case where
every component is a single node — is **answered by composition**.  On the
shipped weight-10 tier that is **18,290 stored entries instead of 1,253,781
(for the 4,934,371 quivers of the set), 179 kB instead of 8.9 MB**, and coverage
grows rather than shrinks: the new set answers **every quiver whose strongly
connected components have total arrow weight ≤ E, of any rank and any total
weight**, which contains the whole current set.  The same decomposition lets
the spec finder search per component (§5), turns the deferred `S`-axiom
verification (D30, ≈ 150 core-hours) into roughly half an hour on one core
(§2), and turns 207 of the 6,799 `crystalline_only` entries green at once,
leaving **1,000 strongly connected quivers** as the whole of that open problem.
The current tier becomes obsolete (user: *"It may make the current dictionary
very obsolete"*); §6 retires it rather than maintaining it alongside.

## 1. The mathematics

**Conventions, as pinned in the repo.**  Arrow `i → j` iff `B[i][j] > 0`;
`⟨γ_i, γ_j⟩ = B_ij`; `X_γ X_γ' = 𝖖^{⟨γ,γ'⟩} X_{γ+γ'}` (on the pentagon,
`B_12 = 1`, `S = E_𝖖(X_1) E_𝖖(X_2)` has coefficient `𝖖³ + 2𝖖⁵ + …` at
`γ_1 + γ_2`, and the reversed product has `𝖖 + …`, a violation); mutating the
node of current charge `g` sends `g ↦ −g` and every other charge
`γ_j ↦ γ_j + max(⟨γ_j, g⟩, 0)·g` (`s_spec_finding.md` §2.1), which is the
Fomin–Zelevinsky mutation of `B` read off the current charges.

Let `Q` have strongly connected components `C_1, …, C_m` in a source-first
order, `Γ = ⊕_k Γ_k` the matching splitting of the charge lattice, and `Γ_k⁺`
the positive cone of `Γ_k`.  The one fact used throughout is

    ⟨Γ_k⁺, Γ_l⁺⟩ ≥ 0   for k < l,

because every arrow between `C_k` and `C_l` points from `C_k` to `C_l`.

### 1.1 Spec composition — a theorem, unconditional

*If `s_k` is a green (negating) sequence of `C_k` for each `k`, the
concatenation `s_1 ++ … ++ s_m`, node indices relabelled into `Q`, is a green
negating sequence of `Q`, and its spec is `spec(C_1) ++ … ++ spec(C_m)` under
the inclusions `Γ_k ⊂ Γ`.*

*Proof.*  Induction on `m`; it is enough to split `Q = A ⊔ B` with every arrow
between the parts going from `A` to `B` (take `A = C_1`).  While `A`'s sequence
runs, an `A`-charge is only ever updated by `A`-charges, so the `A`-charges
evolve exactly as in `A` alone — same node indices, green throughout — and every
emitted `g` lies in `Γ_A⁺`.  A `B`-charge still equal to `γ_b` has
`⟨γ_b, g⟩ ≤ 0`, so it is never updated.  Once `A` is negated, `B`'s sequence
emits charges in `Γ_B⁺`; the negated `A`-charges have `⟨−γ_a, g⟩ ≤ 0` and are
never updated again, and the `B`-charges evolve as in `B` alone. ∎

This is the source-peel lemma of `s_spec_finding.md` §2.1 with the node `k`
replaced by a set of nodes: `|A| = 1` is that lemma and `|B| = 1` the sink-peel
lemma; the acyclic product-spec fact (`strip_spec`) is the case where every
component is a single node; the mantle theorem is the case of peeling
single-node components off the two ends.  The component version is strictly
stronger: two oriented 3-cycles joined by one arrow have no source and no sink,
so the mantle strip does nothing, yet the concatenation of the two 3-cycle specs
is green.  It also extends the boundary §1.4 there draws — *"deletion after a
source/sink mutation isolates provably; freezing never does"* — from one node to
any set of nodes whose arrows to the rest all point one way.  Two consequences:

* any two source-first orders give the same `S`: components that may be swapped
  have no arrows between them, so their factors commute — D19's argument for
  acyclic quivers, verbatim;
* **the arrows between components enter only through their direction** — their
  multiplicities never appear in the spec.

### 1.2 Axiom composition — a theorem

*If `S_k ∈ 𝓔(C_k)` satisfies the BPS quiver constraint on the cone of `C_k` up
to degree `D`, then `S_1 ⋯ S_m ∈ 𝓔(Q)` satisfies it on the cone of `Q` up to
degree `D`.*

*Proof.*  A cone charge of `Q` splits uniquely as `γ = Σ_k γ_k`, and the
coefficient of the product there is `𝖖^{Σ_{k<l} ⟨γ_k, γ_l⟩} ∏_k (S_k)_{γ_k}`
with every exponent `⟨γ_k, γ_l⟩ ≥ 0`.  If only one `γ_k` is non-zero this is
`C_k`'s own coefficient, and the nodes of `Q` are the nodes of the components.
If two or more are non-zero, each factor lies in `𝖖ℤ[[𝖖]]`, so the valuation is
at least 2: no non-positive part and a vanishing `𝖖¹`-coefficient, which is the
constraint at a non-node charge.  Membership in `𝓔` is closure under products,
and `deg γ ≤ D` bounds every `deg γ_k`. ∎

In the reverse order the exponent is `−⟨γ_k, γ_l⟩` and the constraint fails
already at `γ_a + γ_b` for one arrow `a → b` (coefficient `+𝖖`), so the order is
sharp.  "Satisfies the axioms" identifies the product with `S(Q)` only through
`conj:S-unique` (OPEN, `paper_companion.tex`); at the level the dictionary
stores — green specs — 1.1 needs no conjecture.

### 1.3 Content composition — the crystalline factor engine

*In any total order on the factor pairs that places each component's rays before
those of every later component (the mixed rays anywhere), `bps_factor_spectrum`
returns `Ω = 0` on every charge whose support meets two or more components, and
on the charges of `C_k` the multiplicities of `C_k` alone, in the induced
order.*  So the crystalline
`S` of `Q` is `∏_k S_cryst(C_k)`, and its factor content is the disjoint union of
the components'.

*Why.*  The engine reads `Ω` off degree by degree as the palindromic content
that gives the prescribed leading data.  At a mixed charge, with the lower mixed
`Ω` already zero, the partial product's coefficient is the one of 1.2, of
valuation at least 2.  A non-zero palindromic `Ω(𝖖)` has a term at `𝖖^{−m}`,
`m ≥ 0`, and its factor contributes `−𝖖 Ω(𝖖)/(1 − 𝖖²)`, which has a term at
`𝖖^{1−m}` with `1 − m ≤ 1` — forbidden at a non-node charge.  So `Ω = 0` is
forced; and at a pure charge only that component's factors contribute.

*Measured* (part 5 of the script, cone depths 4–6), on five quivers that are
not strongly connected:
3-cycle → node, node → 3-cycle with a double arrow, **Markov → node** (a
`crystalline_only` component), 3-cycle → 3-cycle, node → 3-cycle → node.  In every
case `Ω` vanishes on every mixed charge, each component's own `Ω` is reproduced,
and `S` equals the `lex`-order `S`.  The controls do populate mixed charges —
the reversed component order on 8–31 of them, the `lex` order on 7–16 (and on
none for node → 3-cycle, where `lex` happens to respect the components) — so the
zero is not vacuous.  Comparing with the `lex`-order `S` uses the engine's order
independence, measured throughout the repo and part of `conj:pbw`; in the
component order itself 1.3 needs nothing further.

## 2. What it does to the shipped tier (measured 2026-09-23)

| weight `e` | stored today (cyclic) | strongly connected |
|---:|---:|---:|
| 0 | 1 | 1 |
| 3 | 1 | 1 |
| 4 | 4 | 2 |
| 5 | 29 | 6 |
| 6 | 235 | 23 |
| 7 | 1,859 | 96 |
| 8 | 15,308 | 481 |
| 9 | 128,521 | 2,586 |
| 10 | 1,107,823 | 15,094 |
| **total** | **1,253,781** | **18,290** |

* By rank the strongly connected quivers are: the one-node quiver, 42 at rank 3,
  654 at 4, 4,211 at 5, 8,182 at 6, 4,514 at 7, 662 at 8, 23 at 9, and at rank
  10 the oriented 10-cycle alone.  So 98.5 % of the stored entries, and 99.6 % of
  the 4,934,371 quivers, would be computed rather than stored.  The typical
  stored entry today is one 3-, 4- or 5-node strongly connected component with
  single nodes attached:
  the commonest component profiles are `(3,1,1,1,1,1)` 231,419 times,
  `(3,1,1,1,1)` 176,077, `(3,1⁶)` 147,297, `(4,1,1,1,1)` 112,624.
* **Size.**  The strongly connected entries alone, in shard format 2: **179 kB**
  gzipped against 8.90 MB (× 50).
* **Coverage, checked on every entry.**  Each cyclic component of each of the
  1,235,491 stored quivers that are not strongly connected is in the strongly
  connected set — none is missing.  1,229,899 of them are built from components
  that all carry a spec (today 1,229,692 of those have a spec and 207 are
  `crystalline_only`); the other 5,592 contain a `crystalline_only` component, and
  every one of them is `crystalline_only` today as well — no quiver in the tier
  has a spec while one of its components lacks one.
* **Controls** on a deterministic sample of 2,513 stored quivers that are not
  strongly connected: the composed spec replays green 2,513/2,513; the reversed
  concatenation fails 2,513/2,513; `S` of the composed spec equals `S` of the
  stored spec at depth 3 on 20/20.
* **`crystalline_only`** (6,799): **1,000** are strongly connected (852 of weight
  10; by rank 13 / 137 / 447 / 319 / 65 / 1 / 17 / 1 at ranks 3–10); 5,592 are
  built on one of those; **207 have components that all carry green specs**, so
  their composed spec is green and they are upgraded now.  All 207 are of weight
  10, a 7- or 8-node strongly connected component plus one node, which is then
  a source or a sink — the
  single-node peel lemma already reaches them, and the builder never applied it.
  Those 1,000 strongly connected quivers are the whole remaining problem, and
  they include builder
  misses rather than absences: the oriented 9- and 10-cycles are both
  `crystalline_only@6` in the tier, yet they are mutation-equivalent to `D_9`
  and `D_10` (finite type), and `find_negating_sequence(strategy="auto")` finds
  green sequences of length 16 and 18 (= 2n − 2, as for the 6-, 7- and
  8-cycles) in about 12 s and 55 s — the weight-10 build gave the BFS 30 s.
* **Spec length.**  On 3,761 sampled quivers that are not strongly connected,
  the composed spec is as long as
  the stored one on 2,282, shorter on 449, longer on 1,030.  Every one of the
  1,030 closes once each strongly connected component carries the shortest spec
  the BFS finds (204 components,
  9 s; the edge-multiplicity pruning switched off where it hides the shortest):
  1,007 equal, 23 shorter, none longer.  So the gap is the components' shipped primary
  specs, which were chosen by producer rather than by length, and not the
  composition: with each component's spec at its shortest, a composed spec is never
  longer than today's on this sample, and it is shorter on at least one quiver
  in eight (472; a lower bound, as only the 1,030 were recomputed).
  User: *"who cares if we find slightly bigger specs, they are more principled
  and we can always shorten afterwards if we care about that."*
* **The deferred verification (D30).**  By 1.2 and 1.3 only strongly connected
  entries need the pass; the rest inherit it.  The depth-5 check on 24 sampled
  strongly connected weight-10 entries (ranks 4–8): 24/24 pass, about 0.1 s each
  (0.0–0.3 s).  14,242 such entries carry a green spec, so the pass is **roughly
  half an hour on one core**, where D30 estimated ≈ 150 core-hours for the
  weight-10 set as stored and coded today (a comparable ≈ 0.13 s per entry; the
  count is what changes).
* **Lookup.**  Composition from an in-memory index of the strongly connected
  entries: 0.13 ms per quiver.  `lookup_enumerated` today: 73–82 ms across runs, because it
  parses the whole cell shard on every call.

## 3. The re-designed dictionary

### 3.1 The stored unit, and the completeness statement

**Stored:** one entry per strongly connected BPS quiver up to node permutation,
with total arrow weight `e ≤ E`.  The entry keeps shard format 2's shape —
`q` the arrow string, `s` the node-index string of a green spec (or `v`
explicit charges, or neither with `crystalline_only@D`), `c` the record — in
cells `(n, e)` with self-describing headers and a manifest, as now.  The
one-node quiver is the unit of composition and needs no entry.

**Completeness:** *every strongly connected BPS quiver with total arrow weight
≤ E, up to node permutation* — at `E = 10`, 18,290 quivers in 179 kB.

**Coverage, derived:** *every BPS quiver — connected or not, of any rank and any
total weight — whose strongly connected components have total arrow weight ≤ E.*
It contains the current set, since a component weighs at most what the quiver
weighs, and it is infinite, since the arrows between components are
unrestricted (1.1).

### 3.2 Lookup — one code path

Decompose (Tarjan's algorithm, linear in the size of `B`); for each
component in source-first order, a single node contributes its charge, and
otherwise the component's canonical form is looked up in the in-memory index and
its spec, re-expressed in the component's nodes, is embedded in `Q`'s; the
pieces are concatenated.  An acyclic quiver is the all-single-nodes case, so
D19's acyclic branch and `strip_spec` become special cases of this one path.
The loader's shape is kept — `lookup_enumerated(B) → (entry, spec)` in the
caller's node order — with the entry recording the components it was composed
from.

### 3.3 The record of a composed entry

Read off the components' records, by §1:

| the components | the composed record | licence |
|---|---|---|
| every one green | `green` | 1.1 |
| every one `…axioms@D_k` | `…axioms@min_k D_k` | 1.2 for the leading data; 1.3 for the crystalline agreement (in the component order; in `lex` order through the engine's order independence) |
| some one `crystalline_only@D` | `crystalline_only@min D`, content the union of the components' in the component order | 1.3 |
| some one `…axiom_mismatch@D` | that mismatch | it belongs to the component |

A single node contributes `E_𝖖(X_γ)`, which satisfies the constraint exactly, at
every depth.

### 3.4 Iteration

The stored entries are the strongly connected ones.  Iterating over every
connected quiver of weight `≤ E` stays available by re-enumeration, as the
coded acyclic half is regenerated today (`include_acyclic=True`), each quiver
answered by composition.

## 4. The builder

* **Targets: the strongly connected quivers only.**  For `E ≤ 10`, filter the
  existing enumeration (its stage-0 cache exists); the census of part 1 is the
  cross-check.
* **Climbing in weight (D35, ruled):** a walk over strongly connected quivers
  directly.
  Base: the directed cycles of length ≥ 3.  Moves: raise a multiplicity; add an
  arrow between two nodes with `b_ij = 0`; add an *ear*, a directed path through
  `k ≥ 1` new nodes between two existing ones (`k ≥ 2` when they coincide).
  Completeness: lowering multiplicities reaches the underlying oriented graph,
  and a strongly connected oriented graph has an ear decomposition starting from
  a directed cycle.  Cross-check: the part-1 census, cell by cell, for `e ≤ 10`.
  The strongly connected count grows × 5.0, × 5.4, × 5.8 at weights 8, 9, 10
  (the full set × 7.4), so, if the ratio keeps rising as it has, ≈ 0.1 M at
  weight 11 and ≈ 0.6 M at 12: everything strongly connected through weight 12
  is about the size of the full weight-9 tier.  A projection: the weight
  reached is set weight by weight by footprint and measured build cost.
* **Producers**, unchanged in kind.  Propagation reads a neighbour's spec by
  composition: a target `T` can get a spec by necklacing (T12's local-move
  rewrite brings `k` to the head or tail) from any mutation neighbour `μ_k T`
  whose components are known, so the closure runs over an
  implicit, unbounded set rather than over the stored set plus a tail capped at
  5,000 (D32: the tail discarded 12,764,738 higher-weight quivers that had
  specs).  Then the BFS on the target alone, the order search (skipped at weight
  10, D28), and the crystalline content.
* **The spec kept for a strongly connected entry is the shortest known**, since
  every quiver containing it as a component reuses it; shortening further stays optional
  (user, D39).
* **The verification pass (D30)** runs on strongly connected entries only.
* **The class index** (mutation components) is not part of the store: the rich
  build keeps it, and mutation orbits belong to the theory dictionary of D15.

## 5. The spec finder — `find_negating_sequence`

**IMPLEMENTED 2026-09-23 (D37, T37).**  `find_negating_sequence(…, decompose=True)`
under both strategies; the `lookup` seam per component, under `"bfs"` too;
`decompose=False` the pre-D37 search, kept as the cross-check.  What moves and
what stays whole for `BPSKAlgebra` and the other consumers is T37's record in
`tasks.md`; the measurement is `experiments/strongly_connected_spec_finder.py`.
The same decomposition now sets the crystalline factor engine's default order on
a cyclic quiver that is not strongly connected (D44: each component built alone,
the product assembled source-first — §1.3 made the build).

User: *"true, this hits even find_negating_sequence!"*

Today `strategy="bfs"`, the default, searches the whole quiver;
`strategy="auto"` (`find_spec_auto`, which `CoulombAlgebra` in
`bps_quiver_tools` uses for its auto-find since the 2026-07-16 ruling) splits the
quiver into decoupled components, strips the mantle, searches the core, and has
an optional `lookup` seam.

**Proposed:** the strongly connected decomposition goes first under every
strategy — it subsumes both the decoupled-component step and the mantle strip —
and then, per component, the dictionary through the existing `lookup` seam and,
only for a component outside it, the chosen search on that component alone; the
pieces are concatenated (green by 1.1) and replay-verified as now.  The search
cost then scales with the largest component instead of the rank; the plain BFS
grows about 3–4× per unit of rank (`s_spec_finding.md` §1.1).  The decomposition
is over the mutable nodes, as `find_spec_auto`'s is today.  Call sites that gain:
`CoulombAlgebra`'s auto-find (`bps_quiver_tools.py:3853`),
`find_spectrum_generator` (`:3266`), `bps_quiver_dictionary` (581, 690, 1224),
`tools/sage_oracle`, the builder's BFS stage, and the applet (Plan 39), where
the whole strongly connected set fits in 179 kB.

## 6. Migration — retiring the current tier

1. **Re-encode `dictionaries/enumerated/` from the shipped tier alone** (no rich
   build needed): keep the 18,290 strongly connected entries with their specs and
   records; upgrade the 207; the manifest keeps, per cell, the census of every
   connected quiver (count, acyclic, strongly connected, composed) as a record
   and a cross-check.  Equivalence, entry by entry, as
   `experiments/shard_format_2_equivalence.py` did for D19: every one of the
   4,934,371 quivers answered by composition, green wherever it was green, `S`
   equal to the old spec's at depth 3 on a sample per cell, and a list of every
   entry whose composed record is shallower than its old one.
2. **Loader:** the one composition path; `acyclic_entry` and the acyclic branch
   become wrappers; `iter_enumerated_entries` yields the strongly connected
   entries and, with the flag, the full census by re-enumeration.  User,
   2026-09-24 (verbatim): *"The dictionary will have a method assembling specs for a general quiver from its strongly connected components if their specs is known"* — the reply to whether `BPSKAlgebra`
   should pass a library `lookup` through to the spec finder.  The assembly
   step exists: `spec_from_components(B, component_spec)` (D38), which
   returns `None` as soon as one component's spec is unknown.
3. **The spec finder** (§5).
4. **The builder** (§4): strongly connected targets; the D30 pass on 14,242
   entries; the 1,000 strongly connected `crystalline_only` quivers as the open
   list, the oriented 9- and 10-cycles
   first.
5. **Consumers to adapt:** `restructuring_plans/42_paper_claims_battery/checks_rg.py`
   samples `iter_enumerated_entries()` at rank ≤ 4, and the stored set changes
   meaning under it; `tests/test_enumerated_dictionary.py` (the census
   assertions of `test_committed_shards`, the acyclic tests); the experiments that
   read shards.
6. **Docs:** CLAUDE.md's dictionary paragraphs, `README.md`,
   `dictionaries/enumerated/README.md`, this plan's README.  The sentence on the
   tier in `paper_companion.tex` is the user's text: an edit will be proposed,
   not made.

The old shards leave the working tree when the new tier lands; git history
keeps them.

## 7. Decisions

All RULED on 2026-09-23 (`decisions.md`): **D33** the stored unit and the
completeness statement — adopted; **D34** the tier stays at
`dictionaries/enumerated/` in a shard format 3 — adopted; **D35** `E` — the tier
**climbs in weight** (the user reversed the recommendation to stay at 10);
**D36** the record of a composed entry — adopted; **D37** the spec finder
decomposing first under every strategy — adopted, its effect on `BPSKAlgebra`
handled first; **D38** names — `strongly_connected_components(B)` in
`quiver_enumeration.py` and `spec_from_components` in `spec_acceptance.py`.
**D39** records the rulings made while the proposal was drafted: the re-design
is unflavoured, and composed specs are acceptable even when longer.

## 8. Out of scope

* **The flavoured tiers** (user: *"focus on unflavoured."*).  One measurement,
  made before that instruction, is kept for whoever picks it up.  On the `SU(2)`
  tier's 27,522 stored entries, 669 are strongly connected.  Of the rest, the
  doublet is two single-node components in 23,089 and lies inside one strongly
  connected component in 3,764, and it is never split between the two, because
  every permutation of an orbit fixes all other nodes.  The composed unfolded
  spec replays green in 26,609 of 26,609 cases where the components have specs.  In
  67 cases the component holding the doublet has a second doublet of its own, so
  a flavoured table of strongly connected quivers would have to key on the orbit
  as marked.
* **Mutation classes** — the theory dictionary of D15.
