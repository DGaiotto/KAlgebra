<!-- Measured report produced 2026-09-22 by a 15-agent workflow (7 entries measured,
each verdict adversarially re-run by an independent agent, then synthesised).
Every verdict in it is backed by a command run on checkout b08eae2; the synthesising
agent states where it relied on a verifier's run rather than its own.
It answers Plan 42 ruling R9 (cover every ADE finite-type class in the companion's
Section 2 extended examples) and the user's requirement that "full examples need a
full trace machinery, possibly involving bootstrap" (2026-09-22).
Findings that are REPAIRS or ERRATA are carried into repo_audit.md section C.errata;
this file is the measurement of record behind them. -->

# Trace machinery for the companion's finite-type examples — measured plan

**Scope.** Every verdict below is backed by a command I ran in this session on this checkout (`b08eae2`). `git status --porcelain` is empty before and after: nothing was edited. Where I rely on the verifier reports rather than my own run, I say so explicitly.

**Box caveat.** 4 cores, shared with two foreign CPU-bound jobs from another session's scratchpad (`…a0eb…/k4.py`, `bpsctrl.py`, 42 and 31 min CPU). Wall times are upper bounds; ratios are robust.

---

## 0. Two corrections to the brief, established by measurement

**`supported_ids()` has 10 entries, not 9, and the zoo has 14 distinct classes behind 17 keys.**

```
supported_ids: ['a1d3','a1d5','a1d7','a3','a5','a7','e6','e8','heptagon','pentagon']
zoo keys: 17 ; DISTINCT classes: 14
aliased: [['a3','hexagon'], ['a5','octagon'], ['a7','decagon']]
```

`hexagon is a3 → True`, `octagon is a5 → True`, `decagon is a7 → True` — *the same class object*. Each hard-codes its base id (`implementations/finite_a3_kalg.py:196` `return trace_residual('a3', …)`; likewise a5:735, a7:4387), so the aliases trace **identically** to their bases (verified: `Tr(1)` and `Tr(L_0)` string-identical at K=12/12/10). The three "unfrozen" polygon entries were never a trace gap.

**The real gap is elsewhere, and it is a correctness gap, not a coverage gap.** Two *frozen* tables silently serve wrong values inside their own served window. That is the finding that should drive the ordering.

---

## A. Rows servable today, by exact call

The companion presents one example per row, so a row counts as servable if one presentation is right.

| # | Table row | Presentation to use | Call | Cost | Safe ceiling |
|---|---|---|---|---|---|
| 1 | `[A₁,A₂ₙ]` trivial | `pentagon`, `heptagon` | `FINITE_KALGEBRAS['pentagon']().trace(A.identity(), K)` | K=64 0.00 s; K=128 1.08 s | none found |
| 2 | `[A₁,A₃]=[A₁,D₃]` SU(2) | `a3` (or `a1d3`) | same, `'a3'` | K=48 0.03 s; K=96 0.22 s | none found |
| 3 | `[A₁,A₂ₙ₊₃]` U(1) | `a5` / `a7` | same | instant | **a5: K≤13, a7: K≤10** ⚠ |
| 4 | `[A₁,D₄]` SU(3) | `SU3ADKAlg` | `from su3_ad_kalg import SU3ADKAlg` | K=12 0.15 s | — |
| 5 | `[A₁,D₂ₙ₊₁]` SU(2) | `a1d5`, `a1d7` | same | K=40 0.03 s | none found |
| 6 | `[A₁,D₂ₙ₊₂]` SU(2)×U(1) | `A1DevenKAlg(3)` = `[A₁,D₈]` | `from a1deven_kalg import A1DevenKAlg` | K=12 0.02 s | — |
| 7 | `[A₁,E₆]` | `e6` | same | K=12 0.00 s | — |
| 8 | `[A₁,E₇]` | **`E7RGKAlgebra`** | `from e7_rgkalgebra import E7RGKAlgebra` | Tr(1) K=12 85 s; lines K=8 ~7 s | — |
| 9 | `[A₁,E₈]` | `e8` | same | K=6 0.00 s | **K≤6, hard** ⚠ |

**All nine rows are servable today.** But three of them (4, 6, 8) are servable *only* off the zoo standalone, and two (3, 9) carry ceilings.

### Row 4 is already written and already correct

The companion's `\subsubsection{$K_\fq([A_1,D_4])$…}` (`paper_companion.tex:566`, label `comp:su3ad`) is built from `SU3ADKAlg`, **not** from the zoo's `FiniteA1D4KAlgebra`. I reproduced its three printed series exactly:

```
Tr(1) K=6  = 1 + [(1,1)]*q^2 + (1 + [(1,1)] + [(2,2)])*q^4
           + (1 + [(0,3)] + 2*[(1,1)] + [(2,2)] + [(3,0)] + [(3,3)])*q^6 + O(q^7)
Tr(L_(0,1,0,0,0)) = -q + (-[(1,1)])*q^5 + O(q^6)                      # = Tr T_0
Tr(L_(0,0,1,0,0)) = (-[(1,0)])*q + (-[(2,1)])*q^3
                  + (-[(0,2)] - [(1,0)] - [(2,1)] - [(3,2)])*q^5      # = Tr D_0
```

matching `eq:comp-su3-traces` term for term. All 8 T/D seeds are `O(q)`, computed in ≤0.17 s.

### Row 8: E₇ needs no freeze

`E7RGKAlgebra` (the ungauged `[A₁,E₇]` as `[A₁,A₆]`+central-chord RG flow) computes what the companion needs, in seconds:

```
E7RG Tr(1) K=4 [0.07s] / K=8 [5.20s] / K=12 [85.1s]
E7RG Tr(1) K=12 vs Nahm _vacuum_rps('e7',12): ALL ORDERS EQUAL = True ; disagreeing orders: NONE
Tr((((3,0,1),),(1,))) K=5 [0.71s] = -q + [(1,)]*q^2 - q^3 + ([(-1,)] + [(1,)])*q^4 + (-4 - [(2,)])*q^5
Tr((((1,5,1),),(0,))) K=5 [0.24s] = -q^3 + O(q^6)
line K=5 vs K=8 agree on q^0..q^5: True (3/3 labels)
```

This is a genuine second presentation tied to the zoo's own e7 BPS spec through `_vacuum_rps`. **The 12-hour freeze the brief anticipates is not on the critical path for the companion.**

---

## B. What needs work

### B1 — `a5` frozen table is silently wrong inside its own wedge (**highest severity**)

Against the independent BPS-free `OctagonKAlg`:

```
a5 (frozen K=24) vs OctagonKAlg Tr(1) K=24  [0.09s]
  POSITIVE CONTROL polygon q^0 = {(0,): 1} -> OK
  q^15 frozen : {(-3,):-10, (-1,):-45, (1,):-45, (3,):-10}
       polygon: {(-5,):-1, (-3,):-10, (-1,):-45, (1,):-45, (3,):-10, (5,):-1}
  FIRST DISAGREEMENT: 15 ; all bad orders: [15,17,18,19,20,21,22,23,24]
```

The gate does not catch it, and **fixing the gate would not be enough**:

```
a5  K_frozen=24 | wedge(STORED row)=19 | wedge(TRUE row)=16 | FIRST WRONG ORDER=15
    -> UNSAFE: serves wrong values at q^15..q^19
    -> even a CORRECTED wedge (16) is still >= first error 15 -> refreeze required
```

Root cause, `finite_kalgebras/elem_traces.py:743`: `w = max(_flavour_weight(k) for k in row)` reads the **stored** row, so a clipped weight is invisible to its own guard.

**The orbit rows are corrupted too, and lower down.** Against `generate_u1('a5',16)` (itself validated: identity disagreements vs `OctagonKAlg` on q⁰..q¹⁶ = **NONE**):

```
a5 ORBIT rows disagreeing with the bootstrap on q^0..q^16: 8 of 24
   orbit 9   first bad q^14  frozen={...(3,):9}          bootstrap={...(3,):9, (5,):1}
   orbit 8   first bad q^15  frozen={...,(5,):1}         bootstrap={...,(5,):-2}   <- sign, not clipping
per-orbit stored wedge (min/max): 16 20
```

Reachable through the public API, instantly and without warning:

```
a5 PUBLIC API Tr(L_9)  K=13 [0.00s] disagreements vs bootstrap: NONE
a5 PUBLIC API Tr(L_9)  K=14 [0.00s] disagreements vs bootstrap: [14]
      q^14 served   : {(-3,):5, (-1,):26, (1,):34, (3,):9}
      q^14 bootstrap: {(-3,):5, (-1,):26, (1,):34, (3,):9, (5,):1}
```

**Safe ceiling for a5/octagon is K ≤ 13.**

**Fix:** re-freeze. `generate_u1('a5',16)` takes **52.2 s** and is correct across every corrupted order. Cost: minutes.

### B2 — `a7` frozen table: identity safe, two orbit rows wrong at q¹¹

The identity row is safe because the wedge sits *below* the first error:

```
a7  K_frozen=16 | wedge(STORED)=11 | wedge(TRUE)=11 | FIRST WRONG ORDER=14  -> SAFE
```

But the orbit rows are not (`generate_u1('a7',14)`, 254.5 s):

```
a7 orbit rows disagreeing with bootstrap on q^0..q^14: 12 of 65
a7 orbit rows whose FIRST BAD ORDER is AT OR BELOW their served wedge (=> silently wrong): 2
   orbit 21  first bad q^11  wedge 12  frozen={...,(3,): 2}  bootstrap={...,(3,): -2}
   orbit 64  first bad q^11  wedge 12  frozen={...,(3,): 2}  bootstrap={...,(3,): -2}
```

This is a **new finding** — the verifier reports analysed a7's identity row only and concluded "SAFE". The sign-flip signature `+2 → −2` on μ^±3 is the same pathology I confirmed independently on a7's identity row against `DecagonKAlg` at q¹⁴ (`frozen {(-3,):-2,(3,):-2}` vs `polygon {(-3,):2,(3,):2}`), so the two corroborate. *Caveat: for the orbit rows the bootstrap is the only witness; the identity row has two.*

**Safe ceiling for a7/decagon is K ≤ 10.** Above its wedge a7 is correct but slow — measured `K=12 207 s`, `K=13 186 s`, `K=14 280 s`, `K=16 454 s`, all `bad orders vs oracle: NONE`.

**Fix:** re-freeze at K=14, ~4–8 min.

### B3 — the su2u1 tier (`a1d4`/`a1d6`/`a1d8`) inverts the flavour peel

The zoo classes serve a q² coefficient that is not the flavour adjoint. Decisive control, using the **repo's own** branching hom:

```
SU(3) adjoint (1,1), dim 8, via zplus_ring.su3_to_su2u1_hom():
    1 + [(1,-3)] + [(1,3)] + [(2,0)]        SU(2) labels [0,1,1,2]  (contains the TRIPLET)

RESTRICTED a1d4 (a1d4_from_su3ad) Tr(1) q^2 : {(0,0):1, (1,-3):1, (1,3):1, (2,0):1}   MATCHES: True
ZOO finite_a1d4       Tr(1) q^2 : {(0,-2):1, (0,0):2, (0,2):1, (1,-1):1, (1,1):1}     MATCHES: False
```

The zoo's reading has SU(2) labels `[0,0,0,0,1,1]` — **no triplet** — which is impossible for a flavour group containing SU(2). Both readings are 8-dimensional, which is why dimension checks pass over it.

Confirmed from a1d8's **own** BPS data (not by comparison to another theory):

```
_vacuum_rps('a1d8',10), peeled two ways vs A1DevenKAlg(3):
  q^2  AS-SHIPPED(slot0)==Deven: False   SWAPPED(slot1)==Deven: True
  q^4  False / True   q^6 False / True   q^8 False / True   q^10 False / True
  q^2 as-shipped: {(0,-2):1, (0,0):2, (0,2):1}
  q^2 swapped   : {(0,0):1, (2,0):1}
  q^2 A1Deven(3): {(0,0):1, (2,0):1}
```

Single shared code site — `finite_kalgebras/su2u1_trace_bootstrap.py:71` `_tr1_nw_from_series`, whose loop `for (w1,w2), v in terms.items(): per.setdefault(w2,{})[w1] = …` peels **slot 0** as the SU(2) weight — used at `:102` and `:371`. Blast radius is exactly `{a1d4, a1d6, a1d8}` (`REGEN_SPECS` su2u1 tier).

**No fix needed for the companion**: rows 4 and 6 have correct presentations already (below). A fix to the tier is a separate, contained job (one peel site, plus a per-entry direction map).

### B4 — `e7` has no frozen table, and the bootstrap route is not usable

Measured: `e7 in supported_ids() → False`. The shipped `FiniteE7KAlgebra.trace` calls `generate_u1('e7', K)` on **every cold process**. My run of `generate_u1('e7', 4, verbose=True)`:

```
[e7] u1 bootstrap: Tr(1) via Nahm-sum (spec) at K=8 (spine-free) ...
  18 ρ²-orbit reps / 90 seeds
  deep pool: 210 labels (12.0s)
  14/18 reps need pairs
```

— still in the pair-augmentation phase at **31 min 28 s CPU / 3.0 GB RSS** when I stopped it. I therefore **could not reproduce** the "pins all 90 seeds" claim, and whatever its eventual verdict, it is not a route anyone can use. (The verifier reported it raises `_BootstrapUnavailable: e7: inconsistent at k=8` after ~64 min; I did not get there.)

**Freeze cost, grounded.** `_generate_bps` (`elem_traces.py:409`) iterates `elementary_seed_indices`, which for e7 returns **all 90** (`fold_policy('e7') = 'none'`, forced for u1 by `elem_traces.py:159`). Measured per-seed cost, cold: `BPS Tr(seed 0) K=4 [202.9s] = -[(-1,)]*q^3 + O(q^5)`. So:

- as-is, `generate('e7', 4, method='bps')` ≈ **90 × ~203 s ≈ 5 h**;
- with δ-propagation ≈ **18 × ~203 s ≈ 1 h** — and the data already exists: `E7_RHO_DELTA` is present, 90 entries, 41 nonzero, and the u1 bootstrap already uses `Tr(L_{ρ²i}) = μ^{δ(i)−δ(ρi)}·Tr(L_i)` to collapse 90 → 18. `_generate_bps` simply does not exploit it. **This is the cheap win if a frozen e7 is wanted.**

Given row 8 is served by `E7RGKAlgebra` today, the freeze is optional.

### B5 — `e8` is capped at K=6 by a hard memory wall  (RESOLVED 2026-09-23: section K)

```
e8 Tr(1) K=6 [0.00s] = 1 + q^4 + 2*q^6 + O(q^7)
e8 Tr(1) K=7 -> OOM-killed
  oom-kill: task=python,pid=29849
  Killed process 29849 (python) total-vm:10755516kB, anon-rss:10646016kB
```

10.6 GB. The design note's "K≈6 gate" is a memory wall, not slowness. **The E₈ example must be written at K=6** — three nonzero orders of `Tr(1)`. That is thin, and it is the honest limit.

### B6 — capability tags are wrong in both directions

```
a3       cone-frozen=['multiply-fast','trace-exact']
hexagon  cone-frozen=['multiply-fast']
a5       cone-frozen=['multiply-fast','trace-exact']     <- the one that serves WRONG values
octagon  cone-frozen=['multiply-fast']                    <- same class object as a5
e7 / a1d4 / a1d6  cone-frozen=['multiply-fast']
```

`finite_kalgebras/objects.py:289`: `if short_id in supported_ids(): cone_caps.add("trace-exact")` — the tag tracks *has a frozen table*, not *computes exact traces*. So `a5` is tagged exact while silently wrong at q¹⁵, and `octagon` — the identical class — is untagged. One-line fix; do not let it drive example selection.

### B7 — the zoo trace suite does not cover any of this

`tests/test_finite_zoo_traces.py:333` `if sid not in supported_ids(): continue` skips e7/a1d4/a1d6/a1d8 and the three aliases. `a5` is covered at `K=4` (`:268`), eleven orders below its first error. The only su2u1 trace test asserts q⁰ only — exactly the order where the inverted peel and the correct one agree. And the suite **does not complete**: my run reached 9 PASSes and then sat on the a1d4 block for >2 min CPU with no further output, matching the reported hang.

---

## C. Documentation discrepancies (errata)

**C1 — `implementations/finite_a1d4_kalg.py:22-24`.** Claims:

> `TRACE: the su2u1 elementary-trace table is served via finite_kalgebras.elem_traces (Layer-1 + bootstrap, the Plan-29 su2u1 trace programme — the same route #520 used for a1d6/a1d8).`

**Actually true:** there is no frozen a1d4 table (`_frozen('a1d4') → None`; `a1d4 ∉ supported_ids()`), and neither has a1d6 or a1d8. The lazy route does return a value, but the value's flavour content is wrong from q² on — `{(0,-2):1,(0,0):2,(0,2):1,(1,-1):1,(1,1):1}` where the SU(3) adjoint branches to `{(0,0):1,(1,-3):1,(1,3):1,(2,0):1}`. The sentence should say the table is *not* frozen and the lazy su2u1 route is *defective*, pointing at `su2u1_trace_bootstrap.py:71`.

**C2 — `finite_type_kalgebras.md:93-102` and `:629`.** Claims "**e7 — SOLVED**", "the bootstrap pins all 90 e7 seeds (over-determined + consistent)", "the per-seed BPS engine is infeasible: **a single e7 seed trace doesn't finish at K=4**", and a scoreboard reading "frozen traces 9/14 (missing: e7/e8 compute-gated…)".

**Actually true, measured:**
- e7 is **not frozen** — `supported_ids()` has no e7. A "SOLVED" entry with no table and no usable route is not solved.
- "a single e7 seed trace doesn't finish at K=4" is **false**: `BPS Tr(seed 0) K=4 [202.9s] = -[(-1,)]*q^3 + O(q^5)`, cold process, first call.
- "the bootstrap pins all 90 seeds" I **could not reproduce**: >31 min CPU, no return.
- The scoreboard is **stale in two ways**: e8 *is* frozen (K=6), and the count is **10/14**, not 9/14.
- `:102` says "Validated BPS-free vs frozen: … a7; e7 Tr(1)/Tr(seed) exact" — there is no frozen e7 to validate against, which `u1_bootstrap.py:18` concedes.

Note the companion already contradicts the design note: `comp:e6`'s standing line says "*E₇ needs its U(1) flavour bookkeeping (finite_type_kalgebras.md, section 3)*" and points at the section claiming it is solved.

**C3 (additional) — `finite_kalgebras/u1_bootstrap.py:18-20`.** "*e7 (no frozen oracle exists, **BPS can't make one**) generated with all 90 seeds pinned and an over-determined-consistent certificate.*" BPS **can** make one — 202.9 s/seed, measured. Same unreproduced pinning claim.

**C4 (additional) — `finite_kalgebras/elem_traces.py:88-92`.** "*su2u1 entries … are NOT generatable yet … Until an entry is frozen, its traces still work — **exactly** — through the lazy path; they are just slow on first call.*" The first half is right and names the direction-map problem; the last clause is false — the su2u1 lazy path is not slow-but-exact, it is **fast and wrong** (a1d4 Tr(1) K=2 in 0.37 s, wrong at q²).

---

## D. Recommended order, cheapest first

| # | Step | Cost | Unblocks |
|---|---|---|---|
| 1 | **Errata C1–C4** (4 files, prose only) | minutes | Stops the next session trusting "e7 SOLVED" or the a1d4 route |
| 2 | **Re-freeze `a5`** — `freeze(['a5'], 16)`; generation measured at 52.2 s | ~5 min | Row 3 above K=13; removes a silently-wrong table |
| 3 | **Re-freeze `a7`** — `freeze(['a7'], 14)`; generation 254.5 s | ~10 min | Row 3 above K=10; fixes orbits 21/64 |
| 4 | **Fix the wedge guard** (`elem_traces.py:743`) to compute `w` from true weights, and fix the capability tag (`objects.py:289`) | ~1 h | Makes 2–3 durable; stops the class of defect recurring |
| 5 | **Write rows 3, 5, 9** from the frozen tables as they then stand | writing only | `[A₁,A₂ₙ₊₃]`, `[A₁,D₂ₙ₊₁]`, `[A₁,E₈]` (E₈ at K=6) |
| 6 | **Write row 8 from `E7RGKAlgebra`** | writing only | `[A₁,E₇]` — no freeze needed |
| 7 | **Write row 6 from `A1DevenKAlg(3)`** | writing only | `[A₁,D₂ₙ₊₂]` |
| 8 | *(optional)* δ-propagation in `_generate_bps`, then freeze e7 | ~1 h compute + small patch | A frozen e7; faster zoo e7 |
| 9 | *(optional)* Fix the su2u1 peel direction | ~1 day | Repairs the zoo a1d4/a1d6/a1d8 standalones |

Rows 1, 2, 4, 7 need **no work** — 1 and 2 are in the paper, 4 and 7 are already written in the companion.

**Why `A1DevenKAlg(3)` for row 6 and not `(2)`.** Measured at K=4:

```
A1DevenKAlg(1) = [A1,D4]: 8 gens ; Tr(L)=O(q) violations 0 ; orthonormality ok=36 FAIL=0 err=0
A1DevenKAlg(2) = [A1,D6]: 30 gens; violations 0 ; ok=14 FAIL=2 err=39
   FAIL I((((4,1),),0),(((8,1),),0)) = (-[(2,(0,))])*q^-2 - [(4,(0,))] + ...   <- a q^-2 POLE
A1DevenKAlg(3) = [A1,D8]: 88 gens; violations 0 ; ok=53 FAIL=0 err=2 (honest freeze-limit raises)
```

k=2 is the defective member and `tests/test_a1deven_kalg.py` covers only k=1 and k=3. Since the row needs one representative, **use k=3**. It is certified term-for-term through q⁶ against `A1DevenRGKAlgebra(3)` (`test_a1d8_trace_matches_reference`), which sits in `certified_oracle_isos()`, and I matched it to a1d8's own BPS vacuum through q¹⁰ (above).

**Row 4, if you ever want it in the zoo's own ring:** `a1d4_from_su3ad.a1d4_kalgebra()` is `SU3ADKAlg.base_change(su3_to_su2u1_hom())`, over `SU2xU1ZPlusRing`, and I cross-certified the two D₄ routes against each other:

```
order | SU(2)-content equal | full (su2, u1*3) equal
  q^0..q^8 : True / True  (all nine orders)
```

i.e. `a1d4_from_su3ad` and `A1DevenKAlg(1)` present the same algebra, differing only by a factor-3 U(1) normalisation — and both differ from the zoo standalone.

---

## E. What the measurements did **not** settle

1. **`SU3ADKAlg` has 3 off-diagonal orthonormality failures among its 8 T/D generators at K=4** — and they are upstream of `base_change`, since the restriction shows the identical 3:
   ```
   verify_orthonormality over 8 T/D gens K=4: ok=33 FAIL=3 err=0
     FAIL ((0,1,0,0,0),(1,1,0,0,0))  FAIL ((2,1,0,0,0),(3,1,0,0,0))  FAIL ((0,0,1,0,0),(3,0,1,0,0))
   ```
   Each failing pair has **identical traces**. Whether the 4 tiles are 4 distinct canonical elements or 2 with multiplicity is open; the test file's own note about "split cases … map to the same A1D4 label" suggests the label set over-counts. This sits under an example the companion already prints as CERTIFIED, so it is worth resolving.

   **RESOLVED 2026-09-22 — a harness artifact, not a failure of the class.**  Each failing pair is ONE element under two raw labels: a bare generator (`T_i` with `b = 0`, or `D_j` with `a = 0`) lies in two tiles.  `SU3ADKAlg.canonicalise` maps both labels of every pair to the same canonical label (`(1,1,0,0,0) → (0,1,0,0,0)`, `(3,1,0,0,0) → (2,1,0,0,0)`, `(3,0,1,0,0) → (0,0,1,0,0)`); the two labels give identical left and right products with a probe and identical traces; and `I(a,b) = I(a,a) = 1 − 𝖖² + (1 + χ₍₁,₁₎)𝖖⁴ + …` (resp. `1 + (−1 + χ₍₁,₁₎)𝖖² + …` for the `D` pair).  The sweep fed raw labels to `verify_orthonormality`, which compares labels, so the pairing of an element with itself was read as an off-diagonal entry.  Lesson for any orthonormality sweep: canonicalise labels first.

2. **The a5/a7 orbit-row corruption rests on the bootstrap as sole witness.** The identity rows have two independent witnesses (`OctagonKAlg`/`DecagonKAlg` *and* the bootstrap, agreeing). For the orbit rows only the bootstrap disagrees with the frozen table. The sign-flip signature is shared with the independently-confirmed identity defect, but a second oracle would settle it.

3. **The e7 bootstrap's actual verdict.** I stopped it at 31 min CPU in the pair phase. Whether it raises `inconsistent at k=8` (as reported) or eventually succeeds is unresolved *by me*. This does not affect the plan — row 8 goes through `E7RGKAlgebra` — but C2's strongest form awaits it.

4. **`E7RGKAlgebra`'s labels are not tied to the zoo's 90 cone seeds.** Its vacuum is tied to e7's own BPS spec (`_vacuum_rps`, exact through q¹²), and `certified_oracle_isos()['E7']` ties the *flow* to this oracle — but there is no label-level `KAlgebraIso` to `FiniteE7KAlgebra`. A companion example written from the RG presentation prints RG labels `((cone-monomial), (μ-charge))`, not cone labels.

5. **No certified iso `A1DevenKAlg(k)` ↔ the zoo standalones.** Generator counts differ (a1d6: 30 vs 39; a1d8: 88 vs 120); the isos in `implementations/u1a1deven_cone_new_iso.py` are for the *gauged* family. The certification chain runs through `A1DevenRGKAlgebra`, not through the zoo class.  *(2026-09-23: those counts were `A1DevenKAlg.mult_generators` missing its generators that are products of two opposite-charge rays; the complete sets have 39 / 120 labels, the zoo's counts (repo_audit C.errata item 22; that the two sets correspond is not checked).  `u1a1deven_cone_new_iso` is in `legacy/`.)*

6. **e8 beyond K=6 was not characterised** — the K=7 attempt died at 10.6 GB before yielding any bound on what K=7 would cost with more memory.

7. **The a5/a7 re-freeze ceiling is untested.** I verified `generate_u1('a5',16)` and `generate_u1('a7',14)`. The verifier reports a5 raising `_BootstrapUnavailable` at k=24; I did not probe where between 16 and 24 it stops, so the maximum honest re-freeze K is unknown.
---

## F. Ruling R10 — which presentation of each row rests on analytic statements plus bootstrap (2026-09-22)

The user's ruling (`decisions.md` R10, verbatim: *"I only care about examples which do not rely on frozen tables"*, then *"But on analytic statements + bootstrap strategies"*) changes the question of section A from "which presentation serves each row" to "which presentation serves it WITHOUT frozen data".  Measured with `frozen_data_audit.py` (this directory).  Each candidate is built, traced (`Tr 1` and one seed, `K = 4`) and multiplied in a fresh process with four instruments:

- any read of the frozen trace table (`finite_kalgebras.elem_traces._frozen`, `supported_ids`) made to raise;
- every non-Python file the run opens, logged (pickles, json, gz);
- the files whose FUNCTIONS execute, recorded from before the candidate's import (an import alone is not use; an import-time computation is);
- every repo module the candidate loads, scanned for numeric-literal density.  A frozen data module is read by attribute lookup and never runs a function, so density is what exposes it: the frozen modules measure 1.4 (`u1a1aodd_k2_chord_charges.py`), 3.7 (`u1_octagon_mult_table.py`) and 2.4–24.8 (the zoo's `finite_*_kalg.py`) literals per line; the densest CODE any candidate loads is 0.58 (`u1_pgon_layer2.py`, closed-form sign and shift rules).  Threshold 1.0.

**Positive controls, asserted** (the script aborts rather than report if one fails): the zoo `FiniteE6KAlgebra` → FROZEN (it reads `_frozen('e6')`); `OctagonKAlg` → FROZEN (both frozen modules above); a `BPSKAlgebra` pentagon → CHART.  The control earned its keep on the first run: installing the trace guard by a plain import pulled in the `finite_kalgebras` package, whose `__init__` preloads the whole zoo, so every candidate read as FROZEN and the pentagon control aborted the run.  The guard is now installed by an import hook and a module counts only if the candidate itself loads or runs it.  Three earlier passes had their own blind spots (a module-diff defeated by that same preload; imports outside the profiled window; frozen files matched by NAME, which missed `u1a1aodd_k2_chord_charges.py` at 247 literals).

**Result — every row has a presentation with no frozen data and no BPS chart:**

| # | Row | Presentation | Relations rest on | Traces rest on |
|---|---|---|---|---|
| 1 | `[A₁,A₂ₖ]` | `A1A2kKAlg(k)` | a closed-form base table from chord geometry (`A1A2k_plucker_closed_form.base_table_predict`) | closed forms: `T₀ = χ₁(𝖖²)`, `T_a = (−1)^{m+1}𝖖^{−m}(χ_m − χ_{m+1})(𝖖²)` with `χ_s` the `M(2,2k+3)` characters, after the Layer-1 cyclicity reduction |
| 2 | `[A₁,A₂ₖ₊₁]` | `ungauge_u1a1aodd(k)` over `U1A1AoddKAlg(k)` | analytic combinatorics (no pickles, no oracle, since 2026-08-31) | Layer-1 reduction to general-`p` closed-form seeds (`u1_pgon_layer2`) plus the orthonormality bootstrap (`u1aodd_trace_bootstrap`).  **One ingredient is conjectural**: the negative-`n` long-chord closed form is measurably wrong at some `ρ²`-orbit positions, so every value passes a theorem-backed guard, is retried at another representative of the same orbit, or fails honestly |
| 3 | `[A₁,D₃] = [A₁,A₃]` | `A1D3KAlg` | six generators and eight `Z₃`-orbits of printed relations | Layer-1 reduction to `Tr 1`, `Tr T`, `Tr D`, then the closed-form affine `sl(2)_{−4/3}` characters |
| 4 | `[A₁,D₄]` | `SU3ADKAlg` | eight printed relation families (pattern-matched once from `SU3BPSKAlgebra`, certified product-for-product against it) | `Tr 1` = the closed-form Kac–Wakimoto vacuum character of `sl(3)_{−3/2}`; `Tr T`, `Tr D` by the orthonormality bootstrap.  Open: E.1 above |
| 5 | `[A₁,D₂ₖ₊₃]` | `A1DoddRGKAlgebra(k)` (`k = 1` is `D₅`) | a flow into `U1A1AoddKAlg(k)` ⊗ SU(2) with `S_RG = E_𝖖(μL)·E_𝖖(μ⁻¹L)`, `L` the short magnetic-charge-1 chord | row 2's machinery, through the flow |
| 6 | `[A₁,D₂ₖ₊₂]` | `A1DevenRGKAlgebra(k)` | a flow into `A1A2kKAlg(k)` ⊗ U(2) with `S_RG = E_𝖖(μ₁L)·E_𝖖(μ₂L)`, `L = L((1, H−2))` | row 1's closed forms, through the flow |
| 7 | `[A₁,E₆]` | `E6RGKAlgebra` | a flow into `U1A1AoddKAlg(2)` with `S_RG = E_𝖖(X_L)`, `L` the central magnetic-charge-1 diameter | row 2's machinery |
| 8 | `[A₁,E₇]` | `E7RGKAlgebra` | a flow into `A1A2kKAlg(3)` ⊕ U(1) with `S_RG = E_𝖖(μL)`, `L` the central chord `(3,0)` | row 1's closed forms (high order limited by the cone reducer on `(3,0)^N`) |
| 9 | `[A₁,E₈]` | `E8RGKAlgebra` | a flow into `U1A1AoddKAlg(3)` with `S_RG = E_𝖖(X_L)`, `L = (3,0)` | row 2's machinery |

Each flow's UV Dynkin type is certified structurally by its UV Cartan determinant (3, 2, 1 for `E₆`, `E₇`, `E₈`).  The shape of the table is a fact about the repository, recorded without a name: every row is one of the two analytic polygon families — `[A₁,A₂ₖ]` on the `(2k+3)`-gon (`A1A2kKAlg`), and the `u(1)`-gauged `[A₁,A₂ₖ₊₁]` on the `(2k+4)`-gon (`U1A1AoddKAlg`), ungauged for row 2 — one of the two printed `D`-type cases, or a flow into one of the two polygon families whose `S_RG` is one or two quantum dilogarithms of a single chord.  Rows 2, 5, 7 and 9 inherit the conjectural long-chord closed form of row 2.

**Reach of the exceptional flows' vacuum (measured 2026-09-22, cold, one process per `K`).**  `E6RGKAlgebra`: `Tr 1 = 1 + 𝖖⁴ + 2𝖖⁶ + 3𝖖⁸ + 3𝖖¹⁰ + 6𝖖¹² + O(𝖖¹³)` at `K = 6, 8, 10, 12` in 0.4, 1.3, 2.6, 5.7 s — equal term for term to the frozen zoo `e6` table at `K = 12`, which here serves as an independent witness, not as a source.  `E8RGKAlgebra`: `Tr 1 = 1 + 𝖖⁴ + 2𝖖⁶ + 3𝖖⁸ + 4𝖖¹⁰ + O(𝖖¹¹)` at `K = 6, 8, 10` in 1.0, 3.3, 9.1 s — equal to the zoo `e8` at `K = 6`, and PAST that table's ceiling (section A row 9: `K ≤ 6`, the `K = 7` attempt died at 10.6 GB).  So on the exceptional rows the flow is the better route on reach as well as on provenance.

**Not qualifying:** every zoo standalone (`finite_*_kalg.py`: machine-generated cone data plus frozen seeds); `OctagonKAlg`, `DecagonKAlg` and the standalone gauged polygons `U1{Octagon,…}KAlg` (frozen chord charges and product tables); `A1DevenKAlg` (pickles); `A1DnKAlg` as it stood on 2026-09-22 (its trace was a flat quantum-torus trace — `repo_audit.md` C.errata item 10 — and its multiplication unverified for `n > 3`; rebuilt 2026-09-23, section L); `A1D5KAlg` (Layer 2 not derived).

**What the audit does not show.**  It establishes the ABSENCE of frozen data on the paths exercised (build, `Tr 1`, one seed, one product, `K = 4`); a deeper label could reach code the audit never ran.  It says nothing about correctness — that is the battery's job — and it does not decide whether a flow counts as an "analytic statement" in the sense of the ruling; that is the user's call (raised 2026-09-22).

---

## G. Ruling R11 — self-contained, analytic, fully defined traces (2026-09-22)

The user answered section F's open question (`decisions.md` R11, verbatim: *"No, RG flows would be examples in the RG flow section. In sec 2 extended examples I only want to inclue self-contained classes which use analytic definitions of canonical basis in terms of generators with known relations and have fully defined traces."*).  So the flows of section F go to the companion's RG section, and a Section 2 example needs (a) a self-contained class defining its canonical basis analytically from generators with known relations and (b) fully defined traces.  Every finite-type class in `implementations/`, against the two conditions:

| Class | Row | (a) analytic, self-contained | (b) traces fully defined | Verdict |
|---|---|---|---|---|
| `A1A2kKAlg(k)` | `[A₁,A₂ₖ]` | yes (closed-form table from chord geometry) | yes (closed-form minimal-model characters) | qualifies — the paper's own Appendix |
| `A1D3KAlg` | `[A₁,D₃]` | yes (eight printed `Z₃`-orbits of relations) | yes (closed-form `sl(2)_{−4/3}` characters) | qualifies — the paper's own example |
| `SU3ADKAlg` | `[A₁,D₄]` | yes (eight printed relation families, 0.42 literals/line) | **yes, measured**: 147/147 canonical labels (`a + b ≤ 3`, flavour 1, 3, 3̄) at `K = 12`, slowest 0.4 s; four deep labels at `K = 20` in ≤ 2.2 s; control = the printed vacuum head | **qualifies** |
| `U1A1AoddKAlg(k)` | `u(1)`-gauged `[A₁,A₂ₖ₊₁]` | yes (closed-form rules, fitted at `k ≤ 5` and held out) | **no, measured** at `K = 16`: `k = 1` 233/233 clean; `k = 2` 441 clean + 8 rescued by the `ρ²`-orbit retry; `k = 3` 717 clean + 15 rescued + **5 refused** (the sixth powers of the diameter) | fails (b) |
| `A1D5KAlg`, `A1D7KAlg` | `[A₁,D₅]`, `[A₁,D₇]` | no — the "Plücker base table" is hard-coded data at 4.5 and 11.3 literals/line, in the frozen range | no — `trace` raises `NotImplementedError` (Layer 2 not derived) | fails both |
| `A1DoddKAlg(k)` | `[A₁,D₂ₖ₊₃]` | no — a closed-form fast path over the RG engine, which supplies `ρ`, the trace and the dressed products | through the flow | RG section |
| `A1DevenKAlg(k)`, `U1A1DevenConeKAlgebra(k)` | `[A₁,D₂ₖ₊₂]`, gauged | no — pickled tables extracted from an RG oracle | — | fails (a) |
| `OctagonKAlg`, `DecagonKAlg`, `DodecagonKAlg`, `U1{…}KAlg` | `[A₁,A₂ₖ₊₁]` | no — frozen chord charges and product tables | — | fails (a) |
| `HeptagonKAlg` | `[A₁,A₄]` | no — a wrapper around `BPSKAlgebra` | — | fails (a) |
| `A1DnKAlg(n)` | `[A₁,D_n]` | (as of 2026-09-22) multiplication unverified for `n > 3` | no — a flat quantum-torus trace (C.errata item 10) | fails — *rebuilt 2026-09-23 (R18) as a presentation of `A1DoddConeKAlg` on curve labels, section L; its standing is that class's* |
| the zoo `finite_*_kalg.py` | all rows | the cone data is machine-exported — a COMPLETE finite presentation (every generator, cone and cross product), so products are exact; whether that counts as "known relations" is the user's call | **not uniformly frozen — corrected below**: `a1d5`, `a1d7` serve every seed from closed-form admissible characters | see the correction |

**So the only Section 2 extended example that meets R11 today is `[A₁,D₄]`.**  The gauged `[A₁,A₂ₖ₊₁]` class misses (b) narrowly — no refusal at `k ≤ 2` in this sweep, five at `k = 3` — and every rescue rests on the conjectural negative-`n` long-chord closed form being right at another point of the orbit.  Rows that could be brought to R11 by new work: the gauged class (a full definition of the long-chord traces — the orthonormality bootstrap already serves its intermediate chords), `[A₁,D₅]` and `[A₁,D₇]` (an analytic presentation plus the `SU3ADKAlg` pattern for the trace: a closed-form vacuum character — here `sl(2)_{−8/5}`, `sl(2)_{−12/7}` — and the orthonormality bootstrap).  Sweep: `trace_definedness.py` (this directory).

**Correction (same day, after the user restated the goal as "a self-contained (Cone)KAlgebra which can *in principle* answer every question to arbitrary degree of precision").**  The row above first read "frozen seeds" for every zoo class.  That is wrong for `[A₁,D₅]` and `[A₁,D₇]`: `finite_kalgebras/elem_traces.py::_seed_series` serves their vacuum and every elementary seed from the explicit admissible-character combinations of `a1d5_layer2.py` (`sl(2)_{−8/5}`) and `a1d7_layer2.py` (`sl(2)_{−12/7}`) BEFORE any table lookup; the only `_frozen` call on their path is `trace_residual`'s existence check.  Measured with `_frozen` made to report NO table for them: `FiniteA1D5KAlgebra` gives `Tr 1` to `𝖖⁴⁰` in 0.01 s, equal term for term through `𝖖⁶` (the flow's printed window) to the `[A₁,D₅]` flow `A1DoddRGKAlgebra(1)` (`1 + χ₂𝖖² + (1 + χ₂ + χ₄)𝖖⁴ + (1 + 3χ₂ + χ₄ + χ₆)𝖖⁶`), and all 20 single-generator traces to `𝖖²⁴`; `FiniteA1D7KAlgebra` gives `Tr 1` to `𝖖⁴⁰` in 0.26 s and all 42 generator traces to `𝖖²⁴`.  So both answer every product (exactly, from the complete cone data) and every trace (closed form, any order) — they meet the goal as restated, on the reading that a complete machine-exported cone table is a presentation by known relations.  The same reading makes `e6` and `e8` candidates IN PRINCIPLE (vacuum from the exact Nahm sum on the embedded spec, seeds by the orthonormality bootstrap — "fully spine-free and arbitrarily q-improvable" per the code — but `e8` exhausted 10.6 GB at `𝖖⁷`), and leaves `a5`/`a7` (bootstrap measured correct to `𝖖²⁰`/`𝖖¹⁴`, `a5`'s raising at `𝖖²⁴`), `e7` (bootstrap verdict unresolved) and the `su2u1` entries `a1d4`/`a1d6`/`a1d8` (fast and wrong, C.errata item 7) out or open.  **Method note**: `frozen_data_audit.py`'s guard raises on ANY `_frozen` call, including that existence check, so it would have mislabelled these two; asking "does it work with the table removed" (make `_frozen` return `None`) is the right test for "does not rely on a frozen table".

**The E series and the D-odd witnesses, after R12 (measured 2026-09-22).**  R12 (the user: a complete machine-exported cone table counts as generators with known relations) admits every zoo `E` class on relations, so only the traces decide.  Each zoo class was traced with its frozen table made ABSENT (`_frozen` returning `None`), in a fresh process under a 9 GB cap and a 20-minute limit per order, with a profiler confirming that no BPS chart ran:

| Class | Order | Result | Witnesses |
|---|---|---|---|
| `FiniteE6KAlgebra` | `𝖖⁸, 𝖖¹², 𝖖¹⁴, 𝖖¹⁶` | 9, 36, 74, 144 s; `Tr 1 = 1 + 𝖖⁴ + 2𝖖⁶ + 3𝖖⁸ + 3𝖖¹⁰ + 6𝖖¹² + 7𝖖¹⁴ + 11𝖖¹⁶`; all 42 generator traces `O(𝖖)` | the vacuum AND all six seeds equal the frozen table through its whole window `𝖖¹²`; the vacuum equals the `E₆` flow through `𝖖¹⁴` (the flow fails at `𝖖¹⁶`, where its auxiliary gauged class refuses a trace) |
| `FiniteE8KAlgebra` | `𝖖⁶` | **timeout** (the seed bootstrap) | the Nahm-sum vacuum alone is cheap — `𝖖¹⁰` in 6.4 s, `1 + 𝖖⁴ + 2𝖖⁶ + 3𝖖⁸ + 4𝖖¹⁰`, equal to the `E₈` flow |
| `FiniteE7KAlgebra` | `𝖖⁴` | **timeout** (the `u(1)` seed bootstrap) | the Nahm-sum vacuum equals the `E₇` flow through `𝖖¹²` (section A, row 8) |

So `[A₁,E₆]` meets the goal in principle and in practice to `𝖖¹⁶`; `[A₁,E₇]` and `[A₁,E₈]` meet it only in principle — their vacua are cheap, their seed bootstraps do not finish at `𝖖⁴` and `𝖖⁶` in 20 minutes.  The `[A₁,D₅]` and `[A₁,D₇]` closed forms were witnessed independently by the per-seed BPS engine: the vacuum and all four `D₅` seeds equal through `𝖖⁶` (18.6 s), the vacuum and all six `D₇` seeds through `𝖖⁵` (635 s).

**Products of the D-odd zoo classes (measured 2026-09-23).**  The zoo's cone-to-BPS witness `finite_kalgebras.objects.cone_to_bps_iso` compares U(1) weights: it maps a cone label to its lattice charge and carries the coefficient across unchanged, so it cannot read an `SU(2)` character.  Over all 400 generator pairs of `FiniteA1D5KAlgebra`, against the embedded BPS oracle: 300 products carry no `SU(2)` character and all 300 pass; 100 carry one and all 100 fail; no flavour-free product fails and no flavoured one passes — the witness is exact on what it can read, and silent on the rest.  The flavoured products were then compared with the INDEPENDENT hand-written `A1D5KAlg`, through the flavour dictionary (the zoo's `χ_n·L_γ` against the hand-written label that `r_label_decompose` splits as `(γ, n)`): **400 of 400 generator products agree**, and a doublet coefficient demoted to a singlet is caught (negative control).  For `FiniteA1D7KAlgebra` the hand-written `A1D7KAlg` multiplies only its own 28 atomic generators; on the 196 zoo generator pairs inside that table all 196 agree.  (`kalgebra_object('a1d5')` and `('a1d7')` add the abelian witness as their cone-to-BPS iso; it is certifying only on flavour-free products.)

## Addendum — R14 cleanup (2026-09-23)

On ruling R14 (*"frozen data for which we have an actual functional class is
clearly pointless, clean up as you see fit"*) every frozen trace table but
`e8`'s was removed from `finite_kalgebras/elem_trace_data.py`.  Before
removal, each table was compared with its live route, the table masked
(`_frozen` returning `None`), identity and every seed:

| entry | window | live route | seeds | agree | time |
|---|---|---|---:|---|---:|
| pentagon | 𝖖²⁴ | Nahm-sum vacuum + orthonormality bootstrap | 2 | all | 0.1 s |
| heptagon | 𝖖²⁰ | same | 3 | all | 0.1 s |
| `a3` | 𝖖²⁴ | closed-form characters (the table was generated from them) | 7 | all | 0.0 s |
| `a1d3` | 𝖖²⁰ | `su2` bootstrap | 3 | all | 6.7 s |
| `e6` | 𝖖¹² | Nahm-sum vacuum + orthonormality bootstrap | 7 | all | 9.7 s |

`a1d5` / `a1d7` were never read (the closed forms are served first); `a5` /
`a7` were wrong inside their served windows (C.errata items 5–6) and now go
through the `u(1)` bootstrap.  At the tables' full windows the live routes
cost: pentagon 𝖖⁶⁴ 0.1 s, heptagon 𝖖⁴⁸ 11.1 s, `a3` 𝖖⁴⁸ 0.0 s.  `e8` stays
frozen (its bootstrap did not reach 𝖖⁶ in 20 minutes).

**The D₇ product witness.**  The battery row `comp:a1dodd/implementation`
first sampled D₇'s BPS witness on `range(0,42,3) × range(0,42,2)`; the
recording run spent ~15 minutes on its first five pairs and was stopped.
Timed pair by pair (30 s cap, cheapest BPS charges first): the 64 ordered
products of the eight generators whose BPS charge is a single node charge up
to sign (−e₀…−e₅, +e₅, +e₀) are all flavour-free but two, and all agree; 59
take under 10 s, and the three involving generators 28 and 30 took 63 s,
0.0 s (cached) and 473 s.  That set is the witness sample now — deterministic,
and about ten minutes.

---

## H. The u(1)-gauged [A₁,A₂ₖ₊₁] trace: one closed form for every seed (2026-09-23)

The user's request (2026-09-23): *"I am hoping you will finally figure out the powers of \fq etc. which can make U1A1AoddKAlgebra work as well as A1AevenKAlgebra does."*  Section G recorded this class as failing R11's condition (b): the fitted seed forms of `u1_pgon_layer2` refused 5 labels at `k = 3` and needed an orbit retry on 23.

**The rule** (`u1_pgon_layer2.singlet_chord_trace`).  With `p = k + 2`, `r = n + 1`, and `χ̂_{r,s}` the false-theta numerator of the `M(1,p)` singlet module `M_{r,s}` (the `1/η` stripped, in `𝖖`, divided by its leading power — in integers `Σ_{i≥0}[𝖖^{2pi²+2i(pr−s)} − 𝖖^{2pi²+2i(pr+s)+2rs}]`):

    Tr(L_{2j} Eⁿ) = (−1)^{(p−1)n+j+1} · 𝖖^{(p−1−2j)n−j} · (χ̂_{r,j} − χ̂_{r,j+1}),   χ̂_{r,0} := 0,

for every even chord type `2j` (`j = 0` the `E`-tower, `j = 1` the length-3 chord, `j = (p−1)/2` the diameter at odd `k`) and every integer `n` (the same expression continued to `r ≤ 0`); odd chord types carry gauge charge and vanish.  The class's character index is `n = −base`, `base = (−1)^{position}·g₀`, for every type and every `k`.  At `n = 0`: `(−1)^{j+1}𝖖^{−j}(χ̂_{1,j} − χ̂_{1,j+1})` — `A1A2kKAlg`'s `T_a = (−1)^{m+1}𝖖^{−m}(χ_m − χ_{m+1})` with singlet characters in place of the `M(2,2k+3)` ones.

**How it was found.**  An algebraic rewriting of the long-chord form the item-1 scout fitted against the bootstrap: the four partial thetas pair, per module, into `𝖖^{C}(𝖖^{h} − χ̂)`, both pairs share the constant `(p−3)n − 1`, and the stray monomials cancel identically; the same shape, with `s ∈ {j, j+1}` read off the shipped atypical code for type `2j`, then PREDICTED the diameter and the intermediate chords.  Verified identical to the scout's fitted long chord (`p = 3..9`, `n = −6..6`, through `𝖖¹⁶⁰`) and to `tr_v_n` at `j = 0`.

**Two independent exact routes** (sharing nothing but the algebra's axioms):

| route | what | result |
|---|---|---|
| orthonormality bootstrap (trusting only the `E`-tower) | `k = 2` to `𝖖⁸⁴` (43,031 eqs), `k = 3` to `𝖖⁶⁴` (51,843 eqs), 0 inconsistent | 21/21 and 27/27 seeds equal the rule over their determined ranges |
| same, HELD OUT | `k = 5` (`p = 7`) to `𝖖⁴²`, 21,206 eqs, 0 inconsistent | 28/28 seeds of types 2, 4, 6 equal the rule; the diameter's `−𝖖³⁷ + 𝖖⁴¹` was predicted (the scout's fit gave `𝖖³⁵, 𝖖³⁹`; the old two-branch form neither) |
| RG transport `Tr_UV(b) = Tr_aux(ρ(S)Φ(b)S)` down `U1A1AoddToEvenQTRGKAlgebra` into `A1A2kKAlg ⊗ QT` (`experiments/u1a1aodd_trace_transport.py`) | `k = 2` long, 7 seeds to `𝖖⁴²`; `k = 3` long, 14 seeds to `𝖖³²`, diameter to `𝖖⁴⁴`; `k = 4` type 4 to `𝖖³²` | all equal the rule; at `k = 4` it overturns two deep gauged-`A₉` oracle anchors (`−𝖖²⁴`, `+𝖖²⁷`) that the old test file carried |
| the class's own bootstrap (intermediate chords) | `k = 4`, types 2 and 4, positions 0–3, `n = −2..2`, `𝖖²⁰` | 40/40 |
| BPS / oracle anchors in `tests/test_u1_pgon_layer2.py` | long `p = 4, 5`; diameter `p = 5`; `p = 7` low order | 17/17 |

**The gauge of the transport.**  `S = E_𝖖(X_{0,1}L)` depends on the dressing chord `L`; the `E`-tower cannot see the choice, and the long chord only partly (at `k = 3` the chords `i₀ = 3, 5, 7` all reproduce it).  The diameter discriminates: `i₀ = 3` adds a spurious `−𝖖²⁶`.  The chord `{2k+1, 0}`, ending on the merged vertex of `Φ`'s vertex merge, passes every control at `k = 2, 3, 4`.  (An attempt to fix the gauge by the discovery relation `RG(b)·S = L_b + O(𝖖)` failed for every `i₀` as coded — a mis-specification of that test, not resolved.)

**What the old forms got wrong** (first wrong order; both exact routes agree on each): long chord `p = 4` `n = 0` from `𝖖³⁹`, `n = −1` from `𝖖³⁴`, `n = 1` from `𝖖⁵⁶`; `p = 5` `n = 0` from `𝖖⁴³`, `n = −1` from `𝖖⁴¹`; diameter `p = 5` `n ∈ {0, 1}` from `𝖖²⁶`; `p = 7` from `𝖖³⁷`.  The per-`k` modules (`u1_octagon_l_long`, `u1_decagon_l_long`) are correct inside their documented windows (`𝖖³⁵`, `𝖖³⁰`) and were extrapolated past them.  Because the cyclicity reduction multiplies seeds by powers down to `𝖖⁻⁴⁵`, these deep errors reached `K = 16`: **37 labels silently wrong** at `k = 2, 3` (`Tr(L_{2,i}⁵)` at `k = 2` from `𝖖¹⁵`, BPS-confirmed) and 5 refused.

**After** (`7a82dab`): `trace_definedness.py u1aodd` gives 233/449/737 clean at `k = 1, 2, 3`, 0 rescued, 0 refused; against the bootstrap all 1,186 labels at `k = 2, 3` agree; `Tr(L_diam⁶) = 𝖖¹² − 𝖖¹⁴ + O(𝖖¹⁷)` at `k = 3`; the `ρ²`-orbit retry is removed and `repo_audit.md` A64's covariance defect is at 0.  Tests: `tests/test_u1a1aodd_singlet_rule.py` (14 checks incl. a negative control), and the three tests that pinned the defect now pin the contract.

**Standing.**  MEASURED by two exact routes at `p = 3..7`, held out at `p = 6, 7`; not DERIVED.  The rule reads as the gauged analogue of the even family's minimal-model rule; a derivation from the `M(1,p)` module theory (which module each chord inserts, and why the index is `n + 1`) is open.  Under R11 the class now meets condition (b) as a closed form at every order; whether "measured closed form" suffices for the companion is the user's call.

## I. The odd-D family [A₁,D₂ₖ₊₃]: the shipped closed forms hold deep (2026-09-23)

The user's request (2026-09-23), after section H: *"Same for A1Dodd and U1A1Deven."*

**What was already there.**  `A1DoddKAlg(k, presentation="cone")` = `A1DoddConeKAlg(k)` has closed-form products at every `k` (the arc rules of `a1dodd_cone_data`; `k = 0, 1, 2` read decoded tables that the frame-free builder reproduces entry for entry) and a closed-form recipe for every trace seed in `a1dodd_layer2` (the 2026-08-31 window rule for `p = 1`, the ladder form for `p = 0`).  Those recipes had been checked to `𝖖¹²`–`𝖖²⁴` — the depth at which section H's fitted forms still looked right.

**Deep check** (the coverage-checked orthonormality bootstrap `a1dodd_trace_bootstrap.bootstrap`, which raises rather than fabricate when a seed loses equation coverage):

| `k` | theory | run | result |
|---|---|---|---|
| 1 | `[A₁,D₅]` | `bootstrap(1, 40, amax=12)`, 54 s | 4/4 seeds equal `a1dodd_layer2` through `𝖖⁴⁰` |
| 2 | `[A₁,D₇]` | `bootstrap(2, 40, amax=10)`, 137 s | 6/6 seeds (incl. the diameter) equal `a1dodd_layer2` through `𝖖⁴⁰` |
| 3 | `[A₁,D₉]` | `amax=10` and `amax=9` both exceed 7.5 GB (the Layer-1 reduction of high generator powers, in the `SU(2)` coefficient arithmetic) | not deepened: the 2026-08-31 `amax=8` run (full coverage through `𝖖²⁴`, `repo_audit.md` A66) remains the `k = 3` depth |

The companion's printed D5/D7 recipes (as `checks_extra_examples._DODD_RECIPES` transcribes them) equal `a1dodd_layer2` through `𝖖⁴⁰` on all 10 seeds, so this deep check certifies the printed formulas too — unlike the gauged-A long chord of section H, whose printed form is wrong from `𝖖³⁹` (`repo_audit.md` C.errata item 14).

Orthonormality sweep of the class (every pair from the generators, their squares, their `χ₁`-dressings and the terms of 40 generator products, `I(a,b) = χ_{κ_a}χ_{κ_b}δ_{a,b} + O(𝖖)` on the sections): `k = 1` 4,560 pairs, `k = 2` 13,695 pairs, `k = 3` 33,153 pairs, 0 failures; the bootstrap fallback in `_trace_residual` never fired.  It could not fire (`_recipe` never returns `None`), so it was removed; the ladder-law guard still raises on a wrong leading term.

**The recipes in Weyl-symmetrised form** (a restatement, no new content).  Write `ch_s := verma·σ_s(μ)/(1 − μ²)` for the unsymmetrised admissible character of module `s` (so `κ₀ = ch₀`), and `W[f](μ) := f(μ) + f(μ⁻¹)`.  Then `κ_s^sym = W[ch_s]`, `κ_s^anti = −W[μ·ch_s]`, and the `p = 0` pair `κ_s^sym + χ₁κ_s^anti = −W[μ²·ch_s]`.  Index the modules by their valuation `t(s) = min(2s, v − 2s)` (a bijection of `{1,…,k+1}`; `val κ_s = 2t`).  With `ε = a mod 2`:

    Tr(a, 1) = 𝖖^{−a} · Σ_{t=a}^{k+1} W[μ · ch_{s(t)}]
    Tr(a, 0) = 𝖖^{−ε} (ch₀ + Σ_{t even ≤ a} 𝖖^{−t} W[ch_{s(t)}])  +  𝖖^{ε−1} Σ_{t odd ≤ a} 𝖖^{−t} W[μ² · ch_{s(t)}]

so `p = 1` is a tail sum over the modules of valuation `≥ a` and `p = 0` a head sum over those of valuation `≤ a`.  The shifted blocks `𝖖^d W[μ^m ch_s]` (`m ∈ {0,1,2}`, `−4 ≤ d ≤ 0`, `k = 1`) satisfy no integer linear relation through `𝖖³⁰`, so the head sums cannot be rewritten as tail sums; a single-line rule of section H's shape, if there is one, needs different building blocks (open; noted for the user).

**Standing.**  Closed form at every `(a, p, k)`, MEASURED (not derived) and now cross-checked to `𝖖⁴⁰` at `k = 1, 2`.  Under R11 the class meets condition (b).  Open: the default presentation of the factory `A1DoddKAlg(k)` is still `"engine"` (the RG-flow frame); making `"cone"` the default is a one-line change put to the user.

## J. The u(1)-gauged [A₁,D₂ₖ₊₂] trace: an exact route, and a partial rule (2026-09-23, IN PROGRESS)

The same request (*"Same for A1Dodd and U1A1Deven"*).  `U1A1DevenConeKAlgebra(k)` serves its gauge sector `Tr(X₀₁ⁿ)` from Creutzig's closed form (`exact_characters.deven_gauged_xn_qn`) but its matter seeds from a frozen, oracle-read table (`u1a1deven_traces_k{1,2,3}.pkl`: `|m| ≤ 10`, depth `K ≈ 8–12` at `k = 2, 3`; nothing at `k ≥ 4`), and its rays are oracle-extracted ids.

**An exact route** (`experiments/u1a1deven_trace_transport.py`; since 2026-09-23 `implementations/u1a1deven_trace_transport.py`, see the addendum).  Down the flow `U1A1DevenViaDoddRG(k)` into `A1DoddConeKAlg(k−1) ⊗ QT(Z²)` with `S = E_𝖖(X₀₁L)`, the QT trace is `δ_{charge,0}·(𝖖²;𝖖²)²`, so

    Tr_UV(b) = (𝖖²;𝖖²)² Σ_{(x,(0,c)) ∈ RG(b)} Σ_n c_{n+c} c_n Tr_{A1Dodd}(ρ(L^{n+c})·x·L^n),

a finite sum at each order of the A1Dodd closed forms of section I (the stopping rule — three consecutive terms beyond the window — is the U1A1Aodd transport's, empirical).  Control: the gauge tower equals the Creutzig closed form at `k = 1` through `𝖖³⁰` and at `k = 2` through `𝖖²⁴` (`n = 0..3`).  Cost: seconds per seed at `k = 2`, `𝖖²⁰`; minutes at `k = 1`, `𝖖²⁴`; the `k = 2` doublet `(1,1,0)` at `𝖖³²` exceeded 3.5 GB (the A1Dodd reduction of the long words `ρ(L^m)·x·L^n`), so as a trace route it is memory-bound somewhere between `𝖖²⁰` and `𝖖³²` at `k = 2`.  (The flow's own `trace` is the windowed, stability-stopped version of the same sum.)

**The gauge tower as a theta tail.**  With `N(F) := F·(z − z⁻¹)·∏_{j≥1}(1 − z²𝖖^{2j})(1 − z⁻²𝖖^{2j})` (the Weyl numerator; `z` the `SU(2)` fugacity, `p = k + 1`), Creutzig's formula is exactly

    N(Tr X₀₁ⁿ) = (−1)^{np} Σ_{e ≥ n+1, e ≡ n+1 (2)} 𝖖^{G(n,e)} (z^e − z^{−e}),    G(n,e) := p(e² − n² − 1)/2.

The singlet numerators of section H are this sum evaluated at a point: `χ̂_{r,s} = 𝖖^{sr − pr²/2}·Θ_r(z = 𝖖^{−s})`, `Θ_r(z) := Σ_{j ≥ r/2, j ≡ r/2 (1)} 𝖖^{2pj²}(z^{2j} − z^{−2j})`.

**A rule for one kind of matter seed** (MEASURED).  For the chords `(1,0,2)` at `k = 1` (with `n = c₁`) and `(2,1,0)` at `k = 2` (with `n = c₁ − 1`), and — at `k = 1` — `(1,1,0)`, `(1,1,1)` (whose traces are those of `(1,0,2)` at `c₁ ± 1`):

    N = (−1)^{np} [ − Σ_{e ≥ |n|+1} 𝖖^{G+(e−n)} − Σ_{e ≥ m(n)} 𝖖^{G−(e+n)} + Σ_{e ≥ m(n)} 𝖖^{G−(2n+1)} ] (z^e − z^{−e}),

`e ≡ n + 1 (2)`, `m(n) = max(n + 3, |n| + 1)` (at `n ≥ 0` the second and third sums may equally start at `n + 1`: their first terms cancel).  The first two sums are mirror images under `e ↦ −e`, i.e. one signed theta.  Evidence: `k = 1` — `n = 0, ±1, 2` through `𝖖²⁴`, all `n ∈ [−5, 5]` through `𝖖¹⁴`; `k = 2` — `c₁ = 1, 0, −1` through `𝖖²⁰`, with no new parameter, and then HELD OUT: `c₁ = 2, −2, 3, −3` (`n = 1, −3, 2, −4`) predicted and equal through `𝖖²⁰`, 4/4; `k = 3` (`p = 4`, never used in fitting) — the dressing-chord doublet `(3,1,0)` at `c₁ = −2..2` equal through `𝖖¹⁶`, 5/5, with `n = c₁ − 1` the only offset that fits.  For the dressing chord `(k,1,0)` itself the index is `n = c₁ − 1` at every `k = 1, 2, 3` (at `k = 1`, `Tr((1,1,0), c₁) = Tr((1,0,2), c₁ − 1)`).

**Open: the other matter seeds.**  The chords `(1,0,0)`, `(1,0,1)` at `k = 1` and `(1,0,0)`, `(1,1,0)`, `(2,0,0)` at `k = 2` have numerators that are not sums of a few such tails (at `k = 2`, `(1,0,0)|0` carries `𝖖⁷(1 − 𝖖²)² + …` at `z³`; many terms per `z`-power).  They satisfy no integer relation with `𝖖`-shifted copies of the rule above and of the gauge tower in a window `m ∈ [n−3, n+3]`, `|d| ≤ 4` (a larger window was underdetermined and is discarded).  Which kind a chord is depends on its position relative to the dressing chord `L`, not only on its type `(a, p)`.  Put to the user, together with the option of making the exact transport the class's trace route now (every trace fully defined, no frozen box) and the character rule a follow-up.

**Addendum (2026-09-23): the exact transport is the trace route** (R16 item 1, carried out under R19).  `U1A1DevenConeKAlgebra.trace(a, K)`: the section of `a`; magnetic charge `≠ 0` gives 0; no ray factors gives Creutzig's closed form; otherwise one call of the transport on the flow label — `DevenTraceTransport`, promoted to `implementations/u1a1deven_trace_transport.py` (the experiment's gauge-tower control is now that module's `__main__`, and the experiment file is gone: of the same name, it shadowed the module for scripts run from `experiments/`), with `trace_aux(X, K)` for an auxiliary element.  The cone is built on `U1A1DevenViaDoddRG(k)` at every k, so a section is a flow label; the trace pickles and the legacy-built `k = 1, 2` table pickles are deleted.  The stopping rule stays a measured hypothesis, now guarded (a sector also waits until its last four term valuations strictly increase; that extended some labels of two to four letters at k = 2, 3, and the 37 extended results compared with the flow's own windowed trace all agree) and capped (`ValueError`, never a silent truncation).  Evidence: the gauge tower to `𝖖⁴⁰` (k = 1, 2) and `𝖖³²` (k = 3); the old frozen tables reproduced 60/60 (k = 1), 466/466 (k = 3) and 204/210 (k = 2), the six differences exactly rays 20/24 at `X01^{0,±1}` — the frame mismatch of repo_audit C.errata item 17, now closed; orthonormality on every ordered ray pair at `K = 3` (256/256, 2601/2601) and for `A1DevenKAlg` on its complete generator set at `K = 3` (k = 1: 81/81 ordered pairs, k = 2: 1600/1600; k = 3: every diagonal pair and 60 random off-diagonal ones involving a product of two rays, 181/181), with no negative `𝖖`-power anywhere.  The transport's limit on the length of an A1Dodd word (`max_word_degree`) was then 12 letters at every k, which refused labels of high `X01` power at moderate depth (5 of the 736 frozen seeds, all at `𝖖¹²`: `X01` power 12–14 at k = 1, −10 at k = 3; and, found by the review of the same day, 5 of the 8 `A1DevenKAlg(1)` generators at `𝖖¹²`, which the frozen tables had served); since that review the default is per k, set from measured memory (32 / 18 / 16 letters at k = 1 / 2 / 3; `u1a1deven_trace_transport._MAX_WORD_DEGREE`, whose comment has the measurements), which serves all of them, the values unchanged.  Open: the character rule for the remaining seeds (which would also lift that boundary), and the geometric frame.

**Addendum (2026-09-23): the geometric frame, first three steps** (R16 item 2, under R19).  `implementations/u1a1deven_geometric_frame.py` — provisional: no public surface, `U1A1DevenConeKAlgebra` unchanged; tests `tests/test_u1a1deven_geometric_frame.py` (19 tests, about 36 s; `deep` adds k = 3 orthonormality).  Labels are the curves `(x, ℓ)` of `A1DnKAlg` (R18: from marked point `x` to `x + ℓ`, `ℓ` boundary edges on the side without the puncture, `ℓ = n` the loop around it), on the `(2k+2)`-gon with one interior puncture, times a power of `E = X_{0,1}`.
(i) *Curves, for any n* (the `a1dodd_cone_data` helpers taking `n` and a curve): `n(n−1)` curves and `C(2n−2, n−1)` maximal non-crossing sets, all of `n − 1` curves, at every `n = 3..10` (so `C(4k+2, 2k+1)` at `n = 2k+2`); the crossing count equals `a1dn_kalg._arc_crossings` on every pair and, at odd `n = 3..9`, `arcs_cross` / `arc_puncture_crossing`.  A pure-Python search for ρ-equivariant graph isomorphisms finds the curves (non-crossing, rotation) isomorphic to the rays of the Dodd-built tables with the rays that are products of two rays removed (`𝖖`-commuting, the tables' ρ): 16 = 12 + 4, 51 = 30 + 21, 120 = 56 + 64 rays at k = 1, 2, 3 and, held out, 235 = 90 + 145 at k = 4 (rays, ρ and the `𝖖`-commute graph built with cocycles and cross products stubbed — that build reproduces the full one's graph at k = 2); every removed ray is the product of a `+1` and a `−1` ray; `n` isomorphisms at each k, one per rotation; the k = 4 table cones restricted to the kept rays are the 48620 = `C(18, 9)` cones of 9 curves.  Controls at k = 2: one deleted edge, and ρ = rotation by 3, leave none.
(ii) *The flow's RG image Φ and ρ in closed form*: the vertex merge of `U1A1AoddKAlg`'s embedding, one polygon smaller — marked point 2 of the `(2k+2)`-gon removed, the others relabelled `v ↦ (v or v − 1) + S` on the `(2k+1)`-gon — with `S = _ap_base(k, 1, k − 1) mod (2k+1)`, the start of the flow's letter `L = (k, 1, 0)` read as the curve `(S, 2)` (so `L = Φ((0, 3))`): `1, 0, 5, 7, 8` at k = 1..5, read, not fitted.  One term for a curve with no endpoint at the merged vertex, two for one endpoint on it (moved back at `E⁰`, forward at `E¹`), four for the loop at it (the loop at `π(1)` at `E⁰`, `(𝖖⁻¹ + 𝖖)·(π(3), 2k)` and `χ₁` at `E¹`, the loop at `π(3)` at `E²`); in the normalisation "the term of lowest `E`-power has `E`-power 0" that term is the curve's flow label.  Exact against `U1A1DevenViaDoddRG(k).RG` on every curve, 12 / 30 / 56 / 90 / 132 at k = 1..5; `ρ((x, ℓ)) = E^d·(x + 1, ℓ)`, `d = −#`(endpoints at marked point 1), `ρ(E) = E⁻¹`, exact against the flow's ρ on the same curves.  Controls, curves still exact out of 12 / 30 / 56 / 90 / 132: `S + 1` 5 / 14 / 29 / 50 / 77; the two merged-vertex terms at swapped `E`-powers 8 / 22 / 44 / 74 / 112; ρ's drift at marked point 3, 6 / 16 / 34 / 60 / 94.  The magnetic charge is `0` on odd `ℓ`, else `−1` when the endpoints have the parity of the merged vertex and `+1` otherwise, and `UngaugedKAlgebra.mag` is `−2·c0` — the endpoint-parity rule of `U1A1AoddKAlg` relative to its merged vertex `H − 1` (checked there at k = 1..3).
(iii) *A skeleton with a trace that calls no RG* (`_CurveFrame(k)`, private, no product): labels `(curves, e)`; `Φ(label)` = the product of the `Φ(curve)^m` and `X_{(0,e)}`, scaled so its lowest-`E` term has coefficient 1 (the cocycle normalisation); `trace` = magnetic vanishing, then `DevenTraceTransport.trace_aux(Φ(label))`; the pairing = the same transport on `Φ(ρa)·Φ(b)`, a product in the auxiliary algebra.  Checked with the flow's `RG` disabled.  Evidence: `Φ(label) = RG(flow label)` and ρ equal to the flow's on every curve and 40 random monomials at k = 1, 2, 3; the gauge tower `Tr(Eᵉ)`, e = 0..3, equals Creutzig's closed form through `𝖖⁴⁰` (k = 1, 2), `𝖖³²` (k = 3), `𝖖²⁴` (k = 4), `𝖖²⁰` (k = 5), under a second each, `e + 1` differing; traces equal `U1A1DevenConeKAlgebra`'s through the closed-form label map on 20 / 16 / 10 charge-0 labels at k = 1 / 2 / 3 through `𝖖⁸` / `𝖖⁸` / `𝖖⁶` (18 / 14 / 5 of them nonzero); orthonormality `δ + O(𝖖)` with no negative power on the identity, `E^{±1}`, the curves, their squares and the terms of 40 random products of two crossing curves (taken by the cone class through the label map, 2–4 terms each): every diagonal pair (79 / 153 / 207) and 1062 / 4382 / 13568 off-diagonal pairs at k = 1 / 2 / 3 (a probe run; the default test covers k = 1 in full and a sample at k = 2, `deep` k = 3); without ρ the diagonal fails on every curve, charge-0 ones included, and with ρ's drift at marked point 3 it fails on exactly the 6 / 14 curves (k = 1 / 2) whose image that changes; `ρ²`-twisted cyclicity on 30 pairs of total charge 0 at k = 1, 2, 3 through `𝖖⁶`.
(iv) *Measured, not implemented: the ungauged `A1DevenKAlg`.*  Its generators (8 / 39 / 120 at k = 1..3) are, as multisets of curves, exactly R20's reading: the charge-0 curves (4 / 12 / 24, the odd `ℓ`) and the non-crossing pairs of a `+1` and a `−1` curve (4 / 27 / 96), each once.  Their `E`-powers in the curve frame are not all 0 — {0: 5, −1: 3}, {0: 25, −1: 12, +1: 2}, {0: 83, −1: 28, +1: 9} — because the table build and this frame normalise `E` differently on a few rays (kept rays off `E⁰`: 3 of 12, 3 of 30, 5 of 56, 9 of 90 at k = 1..4).  So on 3 / 14 / 37 generators the `E`-free label of `A1DevenKAlg` is this frame's `E`-free balanced label times a power of the `U(1)` fugacity.
Open: multiplication in the curve frame, the switch of `U1A1DevenConeKAlgebra` to these labels, public names.

**Addendum (2026-09-24): multiplication in the curve frame** (R19, self-contained ADE finite classes with geometric labels; still private and provisional, `U1A1DevenConeKAlgebra` unchanged).  `_CurveFrame.multiply` is the generic cone-monomial reducer over a private cone data — letters the curves and `E^{±1}`, cones the maximal non-crossing sets of curves with `E^{±1}` — with two tables.  *Cocycle*, closed form: for non-crossing curves the `A1Dodd` arc cocycle of their lowest terms (derived: `Φ` is multiplicative and its lowest term is one label with coefficient 1), and `c(curve, E^{±1}) = ±c0`; equal to the peel on every ordered non-crossing pair (72 / 450 / 1568 at k = 1, 2, 3) and on `E`.  *Cross products of two crossing curves*, two routes.  (1) Derived: `Φ(g)·Φ(h)` in `A1DoddConeKAlg(k−1) ⊗ QT(Z²)`, peeled by its lowest-`E` terms with the skein resolutions of `g, h` as the candidates — each lowest term names one candidate at one `E`-power, its coefficient and `χ` are read off, its image is subtracted, and only an exactly zero residual is accepted (otherwise `ValueError`; nothing is fitted).  (2) Analytic: the `a1dodd_skein` model at even `n` — the centrally symmetric skein states for two curves that are not loops (two for one crossing, the Ptolemy resolutions; four for two crossings), the FZ double resolution plus the doubled-diameter `χ₁` daughter for a loop and a curve (that daughter is the two arcs from the curve's endpoints to the loop's marked point; at odd `n` it equals the model's `χ₁` word on every diameter × non-diameter crossing, 6 / 60 / 210 / 504 at k = 0..3), `P² + χ₁PQ + Q²` for two loops; each copy of the boundary edge {1, 2} (marked point 1 to the merged vertex) in a state is one `E` — among rules weighting the boundary edges, commuting with ρ leaves a one-parameter family and this is its measured member — and a loop around the puncture is `χ₁`; the `𝖖`-powers are the model's: the bulk rule `Σ_f c(g, f)` (one crossing, two loops), `±` the loop's bulk value (a loop and a curve, sign by operand order), and `A + 1 − #B` for two crossings, `A` the bulk value of the `χ₁` state (at odd `n` the model's anchor `arc_cocycle(g, h)` equals exactly that, 20 / 140 / 504 non-diameter puncture crossings at k = 1, 2, 3).  Every coefficient is a single power of `𝖖`, `χ` at most `χ₁`.  The analytic rule applies to, and equals the derived peel on, every ordered crossing pair: 72 / 450 / 1568 / 4050 / 8712 at k = 1..5 (at k = 4: one crossing 2400, two crossings 840, loop and curve 360, curve and loop 360, two loops 90); `multiply` uses it, with the derived peel where it does not apply (nowhere measured), and `route="derived"` uses the peel throughout.  Evidence, positive control first: the derived peel reproduces `U1A1DevenConeKAlgebra.multiply` through the closed-form label map on one product of each kind at k = 1, 2 (at `E`-powers 0 and (1, −2)); then `multiply` equals the tables on every ordered generator pair (curves and `E^{±1}`: 196 / 1024 / 3364 at k = 1, 2, 3; the derived route too at k = 1, 2) and on 300 random composite products (100 per k = 1, 2, 3, up to three curves per factor, `E`-powers −2..2, up to 87 terms; the k = 3 hundred in `deep`), 150 of them also equal to the peel of `Φ(a)·Φ(b)` with no reducer; the bar involution and ρ as an automorphism on every generator pair at k = 1, 2 and on 60 composite pairs per k = 1..3, the unit law, associativity on all 2744 generator triples at k = 1 and a window of 4913 at k = 2; held out at k = 4, where no table is built: 200 sampled products (150 of two curves at `E`-powers −2..2, mostly crossing; 50 of two composites) equal `U1A1DevenViaDoddRG(4).multiply` on flow labels, and so do all 8464 generator products (`deep`).  Negative controls: with the `χ₁` fork dropped the analytic rule refuses every two-crossing pair (no anchor) and is wrong on every pair with a loop (36 / 150 at k = 1, 2); `E` on the edge {2, 3} or {0, 1}, and the two-crossing anchor with `#B`'s sign flipped, are caught; a missing candidate (either pure daughter of a loop and a curve) makes the peel raise (12 / 60).  Cost: the frame's products need neither the flow nor the auxiliary algebra on the analytic route — 100 composite products at k = 3 in 0.3 s (the table class: about 190 s), the 200 held-out products at k = 4 in 0.2 s (the flow: 2.0 s, at most 0.4 s a pair); none calls `RG`.  Tests: `tests/test_u1a1deven_geometric_frame.py` (27 tests, about 85 s, of which the multiplication tests about 25 s; `deep` adds k = 5, the k = 3 composites against the tables — about 260 s, all of it the table class — and every k = 4 generator product, 9 s).
Open: switching `U1A1DevenConeKAlgebra` to these labels, and public names — both wait for the user's naming rulings.

**Addendum (2026-09-24): the switch — the curve frame is the public `U1A1DevenConeKAlgebra`** (R21, the user's answer to the session's three proposals; R19).  `U1A1DevenConeKAlgebra(k)` (`implementations/u1a1deven_cone_kalgebra.py`) IS the frame, for every k ≥ 1: the private provisional class is folded into it, and `implementations/u1a1deven_geometric_frame.py` stays as its implementation module (curve combinatorics, `Φ` and ρ in closed form, the product rule and the cone data).  Labels `(curves, e, κ)` — the approved `(curves, e)` with the `SU(2)` weight in a third slot, `L_{(curves, e, κ)} = χ_κ·L_{(curves, e, 0)}`, because the contract's `Element` is the Z-form and the even-D family was returning `RLaurent` coefficients in it (`repo_audit.md` C.errata item 25; listed for the user's veto in R21) — and the accessor `curve(x, ell, e=0, kappa=0)`.  The `E`-power is measured from the edge {1, 2} (the class docstring, with the full-turn argument of R21 item 3; `ρ^{2k+2}(curve) = E^{2c0}·curve` pinned at k = 1..4, the same under every re-normalisation of `E`, and no rule treating every position alike reproduces it at k ≥ 2).  Products: the frame's cone data, the analytic rule (the derived peel where it does not apply; `route="derived"`); ρ closed form; `Tr(E^e)` Creutzig's closed form, every other trace the transport on `Φ(label)` (no `RG`); the pairing the transport on `Φ(ρa)·Φ(b)`, equal to the default multiply-then-trace pairing on 24 sampled pairs.  No RG module is loaded on any of these paths: the transport builds the flow's auxiliary algebra `A1DoddConeKAlg(k−1) ⊗ QT(Z²)` and the closed-form parts `C^m = (L^m, (0, m))`, `c_m = (−𝖖)^m/(𝖖²;𝖖²)_m` of `S_RG` itself (equal to the flow's for m ≤ 12 at k = 1..4, traces unchanged), and imports the flow only for its flow-label entry point `trace(b, K)`; its traces are memoised per process and k, so the gauged instance inside each `A1DevenKAlg(k)` reuses them.  Geometric labels: `geometric_label(letter)` (a curve letter is its curve, `None` on `E^{±1}`, as on `U1A1AoddKAlg`), through which `A1DevenKAlg.geometric_label((F, e, κ)) = (F, e, κ)`.  **Where the tables went**: `legacy/u1a1deven_cone_kalgebra_tables.py` (`U1A1DevenConeData`, the table class and its section iso, and the ray-graph helper `_primitive_ray_graph`), `legacy/u1a1deven_cone_build.py`, `legacy/u1a1deven_dodd_build.py`, `legacy/u1a1deven_cone_derivation.py` and `legacy/u1a1deven_tables_k3.pkl`, with pointer headers; nothing on the spine imports them.  The evidence against them is kept runnable as a legacy regression, `tests/test_u1a1deven_tables_regression.py` (8 tests, about 26 s): the ray ↔ curve isomorphism of (i) at k = 1, 2, 3 and held out at k = 4 with its controls; every table generator product equal through the closed-form label map (324 / 2809 / 14884 at k = 1, 2, 3), 100 composites per k = 1, 2 (k = 3 in `deep`); traces equal on 20 / 16 / 10 charge-0 labels and the gauge tower.  `tests/test_u1a1deven_geometric_frame.py` (26 tests, about 100 s) now compares against the flow `U1A1DevenViaDoddRG(k)` directly (both Z-form): every generator product at k = 1, 2, 3 (the flow costs 0.1 / 0.5 / 1.5 s on them) and 100 composites per k = 1, 2, 3 with `SU(2)` weights (2–4 s per hundred for the flow), so the k = 3 composites no longer wait for `deep`.  **The ungauged `A1DevenKAlg`** follows: labels `(F, e, κ)` with `F` a balanced multiset of curves (R20), in the curve frame's `E`-normalisation, Z-form (`L_{(F, e, κ)} = z^{−e}·χ_κ·L_{(F, 0, 0)}`; the lift of PR #1557 unchanged); `mult_generators` is the closed-form list (the charge-0 curves and the non-crossing `(+1, −1)` pairs, 4 + 4 / 12 + 27 / 24 + 96), which the test recovers from the cones and the `E`-commutator; against the table frame 3 / 14 / 37 of the 8 / 39 / 120 generators moved by a power of the fugacity (`{−1: 3}`, `{−1: 12, +1: 2}`, `{−1: 28, +1: 9}` — item (iv) above, now measured against the retired class); its trace reads the lift, the window sum memoised per section (equal to the direct sum on every sampled label).  Consumers re-pointed: `U1A1DevenSkeinKAlgebra` (a Z-form skein cone data with the central `χ` ray and the torus pair `E^{±1}`; its battery passes, 176 grid products), `scripts/atlas_catalogue_gauged_ad.py` (`--full`: u1a1d4 / u1a1d6 / u1a1d8 multiplicative and strictly ρ-equivariant on 14 / 32 / 58 generators).  Speed (2026-09-24, one process each, same machine): 100 random composite products (up to three curves per factor) take 0.06 / 0.13 / 0.21 s at k = 1 / 2 / 3, against 0.05 / 3.0 / 163 s for the table class; construction is immediate, against 0.2 / 1.3 / 0.4 s (live tables at k = 1, 2, the pickle at k = 3); traces are unchanged (the same transport: 20 / 16 / 10 charge-0 labels through 𝖖⁸ / 𝖖⁸ / 𝖖⁶ in 0.2 / 1.4 / 0.4 s either way); `A1DevenKAlg` is comparable (ten generators traced through 𝖖⁸ at k = 1 in 9.9 s against 8.5 s, through 𝖖⁶ at k = 2 in 7.1 s against 9.5 s; 16 / 20 pairings at 𝖖³ in 4.7 / 4.7 s against 4.0 / 8.3 s — the two generator lists name the same canonical elements up to the fugacity).
Open: the character rule for the remaining matter seeds (unchanged); the export trees still ship the table class and the R-form `A1DevenKAlg` until their re-cut.

**Addendum (2026-09-24): closed forms for every seed; the depth boundary lifted** (R19; this answers "the character rule for the remaining matter seeds" above).  `implementations/u1a1deven_seed_characters.py` (`seed_trace(k, curves, n, K)`; research record `experiments/deven_seed_character_rule.py` — the data windows, the fits, the controls and their counts; tests `tests/test_u1a1deven_seed_characters.py`).  The seeds are the gauged elements the traces of `A1DevenKAlg(k)`'s generators reduce to, one per generator (8 / 39 / 120 at k = 1 / 2 / 3): an odd curve, or a non-crossing pair of a +1 and a −1 curve, times `Eⁿ`.  With `p = k + 1`, `z` the `SU(2)` fugacity, `χ_r(w) = w^r + w^{r−2} + … + w^{−r}` (`χ_{−1} = 0`) and the Weyl numerator of the gauge tower above, `N := Tr(seed·Eⁿ)·(z − z⁻¹)∏_{i≥1}(1 − z²𝖖^{2i})(1 − z⁻²𝖖^{2i}) = Σ_{e≥1} c_e(n)(z^e − z^{−e})`, and `w = 𝖖^e`, `σ = 𝖖^n`:

    c_e(n) = (−1)^{pn} 𝖖^{p(e² − n²)/2} F(w, σ)    on the support e ≥ ν + |n − μ|, e − ν ≡ n − μ (mod 2), and 0 elsewhere,

with the corner `(μ, ν)` of the support and the Laurent polynomial `F`:

* the gauge tower `Eⁿ`: `F = 𝖖^{−p/2}`, corner `(0, 1)` — Creutzig's closed form (the positive control);
* the curve `(0, 2j+1)`, `1 ≤ j ≤ k`, `A = k + 1 − j`: `F = (−1)^{p+j}𝖖^{−A}σ^A(χ_j(w) − σ^{−1}χ_{j−1}(w))`, corner `(1, 1)`;
* a nested pair — the charge −1 curve, of length `2j`, inside the other; `g1` / `g2` the marked points strictly between the two at their starts / ends: `F = 𝖖^{−p/2}·ε·σ^{(g1−g2)/2}·Σ_{i=0}^{j−1}𝖖^{2i}B_{(g1+g2)/2+1+2i}`, `B_r = 𝖖χ_{r+1}(w) + 𝖖^{−1}χ_{r−1}(w) − χ₁(σ)χ_r(w)`, `ε = (−1)^{(g1−g2)/2}`, corner `(0, 1)`;
* a side-by-side pair — the charge −1 curve of length `2a`, the charge +1 curve of length `2b`, `g1` the marked points from the end of the first to the start of the second, `g2` the other gap, `g = g1 + g2`: `F = (−1)^{g/2}𝖖^{−g/2−2}σ^{(g2−g1)/2}χ_{a−1}(w)χ_{b−1}(w)(χ₁(w) − χ₁(σ))`, corner `(0, 0)`.

A seed is first carried to the representative of its ρ-orbit (ρ with its `E`-drift; the trace is ρ-invariant).  The loop around the puncture enters as an arc of the same length, and `F` depends only on the configuration (lengths and gaps), uniformly in k.  On the lattice points of the support each monomial of `F` gives a double sum over `a, b ≥ 0` whose shape resembles the partial sums in Kronecker's identity for Appell–Lerch sums — a resemblance of shape only; no identification with module characters is claimed.  Found from exact transport data, one sequence in `n` per ρ-orbit: the numerator solved on the lattice of the support by a sparsest (L1) exact fit on a window (numpy / scipy, research only), accepted only when integral and equal on held-out orders; rewritten in `w`, `σ` the polynomials are `SU(2)` characters.

*Evidence — MEASURED, not derived.*  Against the transport: 1,583 evaluations at k = 1..5 and 108 at k = 6, none disagreeing (data from `𝖖³⁰` to `𝖖⁶⁰` by k and kind).  Held out from the fits: every curve at k ≥ 3, every nested configuration but the four fitted (all 20 at k = 4), everything at k = 5, 6.  The side-by-side rule was corrected once: its first form, `σ^{g/2}`, read at k = 2, 3 where the gap after the −1 curve is always empty, failed at k = 4 on the two configurations where it is not; the power `σ^{(g2−g1)/2}`, fitted there on a `𝖖⁴⁸` window, held on orders 49–56 and on every k = 5 side pair.  Negative controls, each disagreeing: the curve rule of the other curve type, the `U(1)` flow `A → A + 1`, the curves' corner moved to `(0, 2)`, the nested `B_r` with `𝖖 → 1` in its first two terms, the nested gaps swapped, the first side form (where the first gap is nonzero), ρ without its `E`-drift (off the representative).  The witness is not itself certified at these depths: its stopping rule is a measured hypothesis with a guard, and its A1Dodd inputs, the section-I closed forms, are certified through `𝖖⁴⁰` at A1Dodd k = 1, 2 (D-even k = 2, 3) and `𝖖²⁴` at A1Dodd k = 3 (D-even k = 4) — A1Dodd k = 0 (D-even k = 1) is not in that record beyond the gauge-tower control — while the transport evaluates them far beyond the requested depth (the Layer-1 coefficients of its long words; its docstring).  So the evidence is the agreement of two routes that share nothing but the algebra.

*Wired* (2026-09-24, after the switch).  `U1A1DevenConeKAlgebra.trace`: magnetic charge `≠ 0` gives 0; `Tr(Eᵉ)` Creutzig's closed form; a seed `seed_trace` (then `χ_κ`); every other label (products) the transport.  `seed_closed_forms=False` in the constructor skips the seed step: the witness in the tests (`tests/test_u1a1deven_cone_kalgebra.py::test_seed_traces_from_closed_forms` — every seed at k = 1, 2, `E`-powers −1, 0, 2, `κ` = 0, 1 equal to the transport route; a seed never reaches the transport's memo, a curve squared does).  Measured 2026-09-24, one process, no profile hook: `A1DevenKAlg(k)`, every generator through `𝖖⁴⁰` in 0.2 s / 0.8 s / 2.2 s at k = 1 / 2 / 3 (peak RSS 23 / 43 / 81 MB), equal to the transport route on every generator at `𝖖¹⁰` / `𝖖⁶` / `𝖖⁴`; before, through the transport, `𝖖¹⁶` in 116 s / `𝖖¹²` in 771 s / `𝖖⁸` in 1027 s.  The zoo's `a1d6` / `a1d8` (through `A1DevenKAlg(2)` / `(3)`, `finite_kalgebras.a1deven_seeds`): all 39 / 120 seeds and `Tr 1` through `𝖖⁴⁰` in 1.1 s / 2.9 s after the generator map's discovery (2 s / 25 s); the zoo's Layer-1 reduction on the closed-form seeds equals `A1DevenKAlg`'s transport route on 10/10 composite labels through `𝖖⁸` (a1d6) and `𝖖⁴` (a1d8).  The transport's word-length limit now bounds only the traces of labels that are not seeds and `inner_product` (the transport on `Φ(ρa)·Φ(b)`).  The serving guard's `A1DevenKAlg(k)` and zoo `a1d6` / `a1d8` rows are back in the default tier at the standard windows (`finite_type_kalgebras.md` §9).
Open: the traces of products in `U1A1DevenConeKAlgebra` / `A1DevenKAlg`, and their pairings, still go through the transport (the zoo reaches its composites by Layer 1 over the seeds); a derivation of the rules.

**Addendum (2026-09-24): every trace by Layer 1 onto the seeds** (R19; the open item above).  `U1A1DevenConeKAlgebra.trace` on a label that is not a seed now runs the generic Layer-1 reduction of its cone data — `ConeData.simplify_trace_via_cone_data`, the ρ²-twisted cyclicity and the product rule, the reducer `U1A1AoddKAlg` and the zoo use — on the cone data's `χ`-stripped labels `(curves, e)`, through a small private view of the class (`_ChiStrippedView`: its closed-form ρ, ρ⁻¹ and ρ²-orbit representative on those labels); nothing else is new.  The leaves are traced by the class's own steps: magnetic charge `≠ 0` gives 0, `Eᵉ` Creutzig's closed form, a seed its closed form (a leaf that were not a seed would go to the transport and be counted; it never happened).  The pairing is multiply-then-trace.  `seed_closed_forms=False` keeps the transport route whole — every trace with curves and the pairing on `Φ(ρa)·Φ(b)` — as the witness; the served route no longer imports the transport at all.

*Finding.*  In every measurement the leaves are single odd curves times `Eᵉ` and `Eᵉ`: the reduction takes the (+1, −1) pairs apart too.  So, given the product rule and ρ²-twisted cyclicity, every trace of the algebra follows from the single-curve closed forms and Creutzig's tower.  In particular the pair rules can be checked without the transport: each pair seed, reduced by Layer 1 (no pair is left as a leaf), equals its own closed form through `𝖖⁴⁰` at k = 1, 2 (12/12 and 81/81, `E`-powers −1, 0, 1) and `𝖖³⁰` at k = 3 (288/288).

*Evidence against the transport route* (MEASURED).  Positive control first: `M3²` at k = 1, nonzero at `𝖖², 𝖖⁴, 𝖖⁶, 𝖖⁸` by the transport, is equal.  Composite labels (one to three mutually non-crossing curves with powers, `E`-powers −2..2, magnetic charge 0, not seeds): k = 1: 40/40 through `𝖖¹⁰` and 20/20 through `𝖖¹⁶`; k = 2: 40/40 through `𝖖⁶` and 15/15 through `𝖖¹⁰`; k = 3: 30/30 through `𝖖⁴` and 15/15 through `𝖖⁶`.  Pairings: 30/30 at each k (generators, `E^{±1}` and composites; through `𝖖³` / `𝖖³` / `𝖖²`); `A1DevenKAlg(k)` generator pairings 24/24 at each k at `K = 2`.  Negative control: the reducer handed a ρ without its `E`-drift disagrees with the transport on 11/15 (k = 1) and 13/15 (k = 2) composite labels.  The witness's own caveats are the ones stated above (stopping rule a measured hypothesis; A1Dodd inputs certified to `𝖖⁴⁰` / `𝖖²⁴`).

*Costs.*  On these samples Layer 1 was 10 to 50 times faster than the transport, and over 100 times at k = 1 through `𝖖¹⁶` (0.2 s against 24 s).  `A1DevenKAlg` orthonormality on every ordered pair of the identity and the generators at `K = 3`: k = 2, 1,600 pairs in 11 s (through the transport it took about 15 minutes), and k = 3, 14,641 pairs in 144 s — 0 failures, the first check of the complete k = 3 generator set (before: every diagonal pair and 60 random off-diagonal ones).  The serving guard: `A1DevenKAlg(2)` / `(3)` are back in the default tier, and all 47 rows run there (`finite_type_kalgebras.md` §9).
Open: a derivation of the rules.

## K. [A₁,E₈] and [A₁,E₆]: every seed from W₃ characters — the 𝖖⁶ cap is gone (2026-09-23)

The next-session prompt's item 2 (*"[A₁,E₈] seeds (W₃(3,7)/(3,8) hypothesis to test)"*): settled for **W₃(3,8)**.

**The closed form** (`finite_kalgebras/w3_seeds.py`).  Every seed of `FiniteE8KAlgebra` is a finite `ℤ[𝖖^±]`-combination of the five distinct normalised W₃(3,8) characters `χ_h(𝖖²)`, `h ∈ {0, −1/2, −3/4, −7/8, −1}` (Frenkel–Kac–Wakimoto form, pure Python): `Tr(seed) = Σ v·𝖖^j·χ_h(𝖖²)`, `Tr 1 = χ₀`, integral powers of `𝖖` only.  Eight distinct seed values; the sixteen seeds pair up under ρ (`1 = 2`, `10 = 13`, `8 = 15`, `17 = 27`, `21 = 23`, `5 = 22`, `0 = 24`, `26 = 28`).  The recipes are not short — three to nineteen terms, some with polynomial prefactors — so this is a closed form of the `a1dodd_layer2` kind, not a one-line rule.  Wired into `elem_traces._seed_series` beside the A1Dodd override.

**How it was found and what certifies it** (the item-2 scout, 2026-09-23; `tests/test_w3_seeds.py`, 16 checks with E₆):

| witness | range | result |
|---|---|---|
| the ansatz-free solve of every `𝖖^{≤0}` orthonormality equation on the zoo class's own multiply — pins every seed uniquely | through `𝖖³⁶` | 16/16 equal (fixtures in the test) |
| the frozen BPS table (`elem_trace_data['e8']`) | through `𝖖⁶` | 16/16 and `Tr 1` equal |
| `E8RGKAlgebra` (a flow into `U1A1AoddKAlg(3)`) | vacuum through `𝖖¹⁸`; four seed traces at `K = 10`, matched by value | equal |
| character-engine control: W₂(2,5) = M(2,5) against the Rogers–Ramanujan sums | through `q⁴⁰` | equal |
| negative control: one extra term in a recipe | — | caught by the `𝖖³⁶` fixtures |

The W₃ ansatz was solved against the same orthonormality equations the ansatz-free solve pins, so through `𝖖³⁶` the recipes are FORCED by the axioms, not fitted; beyond `𝖖³⁶` they are the prediction.

**After.**  `FiniteE8KAlgebra.trace` answers at any order (`Tr 1` through `𝖖²⁰` in 0.2 s; single seeds and powers instantly), with leading multiplicativity and `I(a,b) = δ + O(𝖖)` on a window.  B5's memory wall concerned the seed BOOTSTRAP; the class no longer needs it.

**E₆ the same way** (W₃(3,7), weights `0, −3/7, −4/7, −5/7`).  The recipes are short: `Tr 1 = χ₀`; seeds `2 = 6`: `𝖖⁻¹(χ₀ − 2χ_{−3/7} + χ_{−4/7})`; seed `4`: `𝖖⁻¹χ₀ + 𝖖⁻³(χ_{−4/7} − χ_{−5/7})`; seeds `5 = 7`: `𝖖⁻²(χ_{−4/7} − χ_{−3/7})`; seed `0`: `𝖖⁻¹(χ₀ − 2χ_{−3/7} + 2χ_{−4/7} − χ_{−5/7}) + 𝖖χ₀`.  Equal to the frozen table through `𝖖¹²` and to the ansatz-free solve through `𝖖³⁷`–`𝖖⁴⁰` (all six seeds).  **Not switched on**: the companion row `comp:e6/traces` certifies E₆'s Nahm-sum + bootstrap route (it runs the class with the frozen table made absent), so serving E₆ from these recipes would change what that row tests — put to the user (R8).  *Switched on the same day under R16 (item 5): the class now serves these recipes, and `comp:e6/traces` runs the Nahm-sum + bootstrap route as the witness, with the recipes made absent.  The frozen `e6` table cited above was removed the same day under R14 (see the addendum "R14 cleanup" above).*

**Standing.**  MEASURED closed forms (forced through `𝖖³⁶` for E₈, `𝖖³⁷`–`𝖖⁴⁰` for E₆).  E₈ has no companion subsection; one would be a proposal for the user (R8); the companion's E₆ rows (`comp:e6/*`) are untouched.  E₇ is not yet treated this way.

## L. The six-item scouting pass: findings not already acted on (2026-09-23)

A read-only scouting workflow (twelve agents: one scout per item, each report re-run adversarially by a second agent) covered the next-session prompt's six items.  Items 1 (gauged `[A₁,A₂ₖ₊₁]`) and 2 (E₆/E₈ vs W₃) are sections H and K.  The rest, with the verifier's verdicts (the probe scripts lived in the session scratchpad and are not kept; the numbers below are what the verifier re-ran):

**E₇ (`FiniteE7KAlgebra`, U(1) flavour).**  The shipped route fails from a harness defect, not the cone data: `finite_kalgebras/u1_bootstrap._sweep` makes a seed value an unknown only at a reduction's deepest `𝖖`-exponent, and `known()` reads any value never solved there as 0.  On e7 the ρ²-representatives `mg3, mg4, mg21, mg27` are never unknowns, so they are silently zeroed; the strict sweep is "inconsistent at k=2", the non-strict one flags 14/18 representatives free and launches 3906 pair reductions (hours, then a raise).  CONFIRMED.  The same pool with the same Nahm vacuum is consistent under a sweep that never reads an unsolved value as 0, and its values agree with BPS (low order) and E₇-flow witnesses; with ρ-equivariance as the fold every seed series is determined through `𝖖¹¹`–`𝖖¹⁴` (the verifier: sweeping to `Kmax = 18–20` on the cheap pool lifts every series to `𝖖¹⁴`–`𝖖¹⁶` with no new reductions).  The vacuum equals the Bershadsky–Polyakov closed form at `k = −12/5` through `𝖖²⁰` (a literature identity; the Nahm sum stays the in-repo witness).  No closed-form seed rule found.  **Fix proposed, not made:** replace `_sweep`'s deepest-exponent-only unknowns with a sweep that keeps undetermined values symbolic, plus a coverage guard.

*E₇ addendum (2026-09-23, later the same day).*  Three measurements bearing on which route can serve `FiniteE7KAlgebra`'s seeds at depth.  (i) **The probe's "determined" carries a hidden support hypothesis.**  Its determinacy report counts a series order as determined when every value with `|μ| ≤ order + 1` is solved; values outside that window may stay free and are then served as 0.  The pinned series sit well inside it (measured `|μ − c| ≲ order/3`, e.g. seed 5 at `𝖖¹³` spans `μ ∈ [−4, 4]`), so the hypothesis is plausible, but it is a hypothesis, not derived — the same `μ`-support issue that made the naive `_sweep` fix fail at a5/a7 (C.errata item 16).  (ii) **The flow `E7RGKAlgebra` is fast only at low order.**  `Tr 1`: `K = 8` 2.4 s, `K = 12` 54 s, `K = 16` 1400 s (×25 per four orders; the `(1,5,1)` label 0.8 / 15 / 278 s).  cProfile at `K = 12`: 94 % of the time is `cone_data.simplify_trace_via_cone_data` — the ρ²-cyclicity reduction of 466 `[A₁,A₆]` words (≈0.38 s each) arising from the flow's 182 auxiliary pairings `Tr_aux(ρ(L^n)·x·L^m)`, `L` the central chord `(3,0)`.  So a transport of the zoo seeds through the flow (the D-even pattern) is fast at depth only if those chord-power pairings in `[A₁,A₆]` get a closed form.  (iii) **Single Bershadsky–Polyakov module characters do not match the seeds** (0 of the 9 ρ-representative series, sign/`𝖖`-shift/`μ`-shift/`μ`-flip allowed, 60 boundary-level candidates, spectral flow NOT included).  Open: multi-module or spectral-flowed combinations (the E₆/E₈ pattern needed 2–5 W₃ characters per seed).  Route choice for E₇ awaits the user.  *(Ruled the same day, R17: routes 2 and 3.  Route 2 landed: the corrected sweep of `finite_kalgebras/u1_bootstrap.py` pins all 90 e7 seeds through `𝖖¹⁴` under the explicit μ-support hypothesis `|μ| ≤ 4k` (measured support `⌈k/3⌉`), with the controls recorded at `repo_audit.md` C.errata item 16.  Route 3 found closed forms for all nine ρ-representatives — theta-product combinations divided by `(q;q)_∞θ(μ)`, validated on held-out orders to `𝖖¹⁹`–`𝖖²⁰`; their independent verification is in progress and they are not yet served.)*  *(Verified and served the same day, with the vacuum in closed form too: section M.)*

**Gauged / ungauged D-even (`U1A1DevenConeKAlgebra`, `A1DevenKAlg`).**  The `k = 2` pole `I((((4,1),),0), (((8,1),),0)) = −χ₂𝖖⁻² − χ₄ + O(𝖖)` (and two more pairs) is a FRAME MISMATCH in the shipped data: the `k = 2` cone's products and ρ were learned from the legacy flow, but the trace pickle's entries for rays 20 and 24 hold the A1Dodd-flow values; replacing just those two towers with legacy-flow values removes every failing pair.  CONFIRMED.  This contradicts `repo_audit.md` A16 as written (A16 says the legacy oracle returns 0 for those rays; today it returns `−𝖖 + 2𝖖³ − (1+χ₂)𝖖⁵ + …`) — the wording of that erratum is the user's.  The scout recommended rebuilding `k = 1, 2` on the A1Dodd flow as `k = 3` is; the verifier did NOT endorse the frame choice from the diagnosis (the two flows' `k = 2` cones have different fans, 51 vs 48 rays).  Separately: the gauged matter bootstrap `solve_matter` reproduces the oracle at `k = 1` (20/20) and raises "inconsistent at 𝖖⁷" at `k = 2`; the verifier proposes first giving it an honest certificate (enumerate every unknown, flag absent ones free, stop swallowing reduction exceptions), since the `𝖖⁷` inconsistency may be the same silent-zero defect.

**Ungauged `[A₁,A₂ₖ₊₁]` (a5 = octagon, a7 = decagon).**  `ungauge_u1a1aodd(k)` traces every label tried (998 at `k = 2, 3`) and its seeds agree through `𝖖⁴⁸` (a5) and `𝖖²⁰` (a7) with an independent route that never uses the singlet rule (the zoo's exported cone table + orthonormality + cyclicity).  But the verifier found that `UngaugedKAlgebra.trace` windowed its gauge sum at `n ∈ [−(K+1), K+1]` instead of around the label's own E-power, silently dropping terms (`trace(E², 0) = 0`, true value `z⁻²`; 18/625 spurious flavour-charged `𝖖⁰` terms in orthonormality at `k = 2`).  **FIXED 2026-09-23** (window centred at `n = −epow(label)`; `tests/test_ungauge_kalgebra.py::test_trace_window_follows_the_label_E_power`).  The zoo a5/a7 path: the frozen tables were wrong (B1/B2) and were removed the same day under R14 (see the addendum "R14 cleanup" above); the `u(1)` bootstrap that now serves them (`generate_u1`) hits the same `_sweep` silent-zero defect ("inconsistent at k=24": representatives 1 and 3 get no equations from order 23 on) and `_seed_series` then falls back silently to the per-seed BPS engine.  Errata candidates: `exact_characters.creutzig_Aodd_Tr1_z` returns wrong output (uncalled); the misleading "inconsistent" message.
*Done 2026-09-23 (R19, the ADE mandate): the zoo serves a3 / a5 / a7 (and hexagon / octagon / decagon) from `ungauge_u1a1aodd(k)`, a1d3 from `A1DoddConeKAlg(0)` and a1d4 from `SU3ADKAlg`, through generator maps built at runtime; the bootstraps and the a3 closed form stay as witnesses.*  **A-odd** (`finite_kalgebras/aodd_seeds.py`): each ungauged generator's gauged charge, magnetic coordinate dropped, minus `f` times the `E` charge, is matched against `<PRE>_MULT_GENS_LATTICE` (a bijection, 6/24/65); a word `((mg, p), …)` maps to (the union of the multisets, `−Σ p·f`), a coefficient `μ^f` to `E^{+f}`, and traces are read back with `z ↦ z⁻¹`.  Each sign was fixed by a positive control: the other sign choices reproduce 18, 18 or 22 of the 36 a3 products, and reading with `z ↦ z` gets 4 of the 6 a3 seeds wrong.  Generator products agree 36/36, 576/576, 4225/4225; `rho_element` on 6/24/65 generators; the seeds equal `a3_elem_entry` through `𝖖⁴⁸` and `generate_u1` through `𝖖²⁰` (a5) and `𝖖¹⁴` (a7, where it takes 150 s); the zoo's own Layer-1 trace equals the ungauged one on 39 composite labels through `𝖖¹²`; `f + 1` on one generator breaks 8, 44 and 148 products.  In a fresh process a5 / a7 serve `Tr 1` and every seed through `𝖖⁴⁰` in 12 s / 31 s, with no BPS function run and no bootstrap imported.  (`tests/test_u1_bootstrap.py`'s `_A5_UNGAUGED` is a trace-only match, eight a5 seeds on one label; a match on traces alone does not fix the label, the products do.)  **a1d3** (`finite_kalgebras/a1d3_seeds.py`): `A1DoddConeKAlg(0)` rather than `A1DnKAlg(3)`, which delegates its products, ρ and trace to it through a relabelling.  Neither side has charges, so the map is the ρ-equivariant bijection reproducing all 36 generator products — three of the 18 do, all related by ρ, under which the seeds are constant.  Seeds equal the BPS engine (`𝖖⁸`), `generate_su2` (`𝖖¹⁰`) and `A1D3KAlg`'s T / D closed forms (`𝖖⁴⁰`), with `Tr 1` and both seeds through `𝖖⁴⁰` in 0.06 s.  `generate_su2('a1d3', 30)`'s "inconsistent at k=23" is the silent-zero defect in `su2_bootstrap._sweep` (instrumented: a seed with no equation at orders 13–22 is read as 0 by 55 later equations; with the true values substituted the sweep is consistent through `𝖖³²`).  It is off every serving path now and is recorded OPEN as `repo_audit.md` C.errata item 24.  **a1d4** (`finite_kalgebras/a1d4_seeds.py`): `SU3ADKAlg`'s seed bootstrap (`sl3_su3_traces`) was audited first — through `𝖖⁴⁰` no equation reads an unsolved value, the isolating term is strictly deepest at every order, and the low-order equations rest on leading multiplicativity `Tr(D^a) = O(𝖖^a)`.  **The U(1) normalisation:** the zoo's U(1) charge `m` is a third of the branching charge `Y` of `su3_to_su2u1_hom` (`U(1) = diag(e^{iθ}, e^{iθ}, e^{−2iθ})`, `3 ↦ 2_{+1} ⊕ 1_{−2}`; the zoo's fugacity is `μ = e^{3iθ}`), and the zoo's generators are `SU3ADKAlg`'s carrying the extra branching charge `c = 0` on the T-orbit and `−2, +2, −2, +2` along the D-orbit.  Of the 32 ρ-equivariant bijections × offsets in `[−6, 6]²`, two reproduce all 64 products, related by ρ².  `rho_element` 8/8.  `Tr 1` and all 8 seeds equal the zoo quiver's own BPS chart through `𝖖⁶` in the Cartan basis, slot 0 the U(1) charge `m = Y/3` and slot 1 the SU(2) weight (0/8 with the slots swapped — the order C.errata item 7 had backwards).  The zoo's Layer-1 trace equals the branched `SU3ADKAlg` trace on 17 composite labels.  Without the factor 3, or with the offsets moved by 3, 32 of the 64 products fail.  a1d6 / a1d8 still refuse.  **Object layer:** the ungauged polygon was not registered in `kalgebra_object('a3' | 'a5' | 'a7')`.  A stock `KAlgebraIso` to `ungauge_u1a1aodd(k).base_change(μ ↦ μ⁻¹)` fails its own battery for encoding reasons: at a3, forward multiplicativity holds on 22/36 pairs, exactly the 14 products carrying μ in a coefficient fail, and ρ fails on the three generators with nonzero `_rho_delta`.  The registered `cone-frozen ↔ bps` edge fails the same way (22/36, 3/6; extends `repo_audit.md` A.5).  Tests: `tests/test_aodd_seeds.py` (25 checks; `--slow` for the full witness orders), `tests/test_a1d3_seeds.py` (11), `tests/test_a1d4_seeds.py` (11).

*Done 2026-09-24 (R19): a1d6 / a1d8 through `A1DevenKAlg(2)` / `A1DevenKAlg(3)` (`finite_kalgebras/a1deven_seeds.py`); every ADE zoo entry now serves its traces.*  Zoo generator `g` is `z^{c_g}·L_{σ(g)}`, `L_{σ(g)}` one of `A1DevenKAlg(k)`'s E-free generators (`mult_generators()`, 39 / 120), and a zoo coefficient `χ_b·μ^m` is `χ_b·z^{s·m}`; a zoo cone monomial goes to the single label of the product of its letters' images, carrying that product's own `z`-power.  All of it is found at first use and nothing is stored: `σ` by a backtracking search over the ρ-orbits (`[3, 6⁶]` / `[8¹⁵]`) for the ρ-equivariant bijections whose generator products agree with the zoo's in a flavour-blind invariant (per product, the multiset over its terms of the (𝖖-power, coefficient) multisets); then, for each such `σ` and each order of the zoo's two coefficient slots, `s` and the offsets `c_g` as the unique solution, by exact elimination over `Q`, of the linear equations the products and ρ impose (`s·δ_g + c_{ρ(g)} + c_g = −e_g`, `δ` the zoo's `_rho_delta`, `e_g` the E-power of the gauged ρ-image); then the certificate — every generator product term for term, and ρ on every generator.  **Measured:** the zoo's slots are the ring's (slot 0 the SU(2) weight, slot 1 the U(1) charge; with the slots swapped the equations have no solution); `s = ±1`, so the zoo's U(1) unit is the ungauged gauge charge itself, with no factor (a ring isomorphism of `R(SU(2)×U(1))` that permutes the basis can only send `μ` to `z^{±1}`; the ×3 of a1d4 above belongs to the SU(3) branching); every offset is determined.  The certified maps are exactly one map composed with the powers of ρ, `s` alternating in sign — 6 at k = 2, 8 at k = 3 — and the first with `s = +1` in the search order is served.  Products 1521/1521 and 14400/14400, ρ 39/39 and 120/120; found and certified in 3.6 s / 28 s.  **Controls:** `A1DevenKAlg(2)`'s `Tr 1 = 1 + (1 + χ₂)𝖖² − χ₁(z + z⁻¹)𝖖³ + …` equals the flow `A1DevenRGKAlgebra(2)`'s through `𝖖⁸` (and `A1DevenKAlg(3)`'s `A1DevenRGKAlgebra(3)`'s through `𝖖⁶`); the same method at k = 1 serves zoo a1d4 exactly as `a1d4_seeds` does (`Tr 1` and the 8 seeds through `𝖖¹²`), a second route to a1d4, independent of `SU3ADKAlg`; negative controls, all caught by the products at k = 2: the slots swapped (803/1521 agree), `s = 3` or `−1` (709/1521), one offset moved by 1 (1435/1521).  The seeds served through the map composed with ρ (`s = −1`) are the same (39/39 through `𝖖⁶`).  **Traces:** the zoo's Layer-1 trace equals `A1DevenKAlg(2)`'s own trace of the image on 20 composite labels through `𝖖⁴`.  Past the transport's limit on the length of an A1Dodd word a seed raises `NotImplementedError` naming the entry, the seed, the order, the class and the limit (checked with the limit lowered to 3 letters); `Tr 1` has no such limit (its gauged sum is Creutzig's closed form).  `objects.py` tags both `trace-exact`: exact arithmetic within the served depth, on the transport's stopping rule (a measured hypothesis with a guard).  **Served depths** (the `--slow` run, 64 min in all): the zoo's `Tr 1` and every seed, through the zoo's own `trace`, equal `A1DevenKAlg`'s at the transport's served depth — a1d6 through `𝖖¹²` (39/39), a1d8 through `𝖖⁸` (120/120) — and its Layer-1 trace equals `A1DevenKAlg(3)`'s own on 16 a1d8 composite labels through `𝖖²`; a composite label's reduction asks its seeds deeper than the label (the 20 a1d6 labels at `𝖖⁴` ask up to `𝖖⁹`), so a composite trace is served while that depth stays within the seeds'.  In a fresh process a1d6's `Tr 1` and its 39 seeds through `𝖖¹²` run with neither `bps_kalgebra` nor `bps_factor_spectrum` nor any bootstrap imported (and through `𝖖⁶` under a per-call spy, with no BPS function called).  `Tr 1` itself goes to any order: through `𝖖³⁰` in 3.6 s (a1d6) and 32.5 s (a1d8), the map included.  Tests: `tests/test_a1deven_seeds.py` (23 checks; `--slow` for the served depths); `tests/test_finite_zoo_traces.py`'s su2u1 test now asserts that all three entries are served.  *(Same day, after the curve-frame switch — R21, section J's last addendum: `A1DevenKAlg` is Z-form on the curve labels `(F, e, κ)`, so `a1deven_seeds` reads its elements through the flavour lift (`_deven_terms`: the label `(F, e, κ)` is `z^{−e}·χ_κ` on the section `(F, 0, 0)`) and `to_ungauged` returns Z-form elements.  Re-run on those labels: the same map classes (4 / 6 / 8, `s = +1` served), products 64/64, 1521/1521, 14400/14400, ρ 8/8, 39/39, 120/120, the negative controls 803 / 709 / 709 / 1435 of 1521, and `tests/test_a1deven_seeds.py` 23/23.)*

*Done 2026-09-24 (R19): pentagon / heptagon through the closed-form geometric class `A1A2kKAlg(1)` / `A1A2kKAlg(2)` (`finite_kalgebras/aeven_seeds.py`); no zoo entry is served by an orthonormality bootstrap any more.*  They were the zoo's last A-even entries on the trivial-R bootstrap `elem_traces._generate_bootstrap` (`Tr 1` the M(2,5) / M(2,7) vacuum character, the seeds pinned by orthonormality), a route the R19 reading in `decisions.md` records as self-contained; the closed forms are cheap, so they serve.  `A1A2kKAlg(k)` lives on the diagonals of the `(2k+3)`-gon (its letter `(a, j)` is `curve(j, a + 1)`, the diagonal from `j` to `j + a + 1`): products are the quantum Ptolemy relations from chord geometry, and the trace is Layer 1 followed by the M(2,2k+3) Andrews–Gordon characters on the product side.  **The map** (found at first use; nothing stored): the zoo's generators carry charges and `A1A2kKAlg`'s do not, so, as for a1d3, the candidates are the ρ-equivariant bijections — one orbit pairing and one cyclic shift per orbit, 5 at the pentagon and 2·7·7 = 98 at the heptagon.  A zoo word goes to the multiset of diagonals letter by letter, its coefficient unchanged: the flavour is trivial, so no coefficient moves and there are no offsets.  **Certificate:** every generator product term for term (25/25, 196/196) and `rho_element` on every generator (5/5, 14/14).  **Measured:** the certified maps are exactly one map composed with the `2k+3` powers of ρ — all 5 pentagon candidates, and 7 of the 98 at the heptagon (the zoo's orbit of generator 0 goes onto the long diagonals `(2, j)`).  The seeds are constant on the ρ-orbits (odd length), so every member of the class serves the same series, and the object layer's own `match_generators` dictionary is one of the members.  **Controls:** positive — the served `Tr 1` equals the M(2,5) / M(2,7) vacuum character in its theta form (`ad_characters.m2_2np3_character`, independent of the product form `A1A2kKAlg` evaluates) through `𝖖²⁰⁰`; and `Tr 1` and the seeds of all 5 / 14 generators equal the bootstrap's through `𝖖⁶⁴` / `𝖖⁴⁸` (and through `𝖖¹⁰⁰` / `𝖖⁸⁰` in a one-off run where the bootstrap took 0.4 s / 32 s).  Negative — among all 120 bijections of the pentagon's generators exactly the 5 certified ones reproduce the 25 products (the other 115 at most 11); the 91 other heptagon candidates reach at most 84/196; every transposition of two generators fails (at most 11/25 and 144/196); and the reflection of the polygon gets 5/25 and 70/196.  **Traces:** the zoo's Layer-1 trace equals `A1A2kKAlg`'s own trace on 11 / 71 composite labels through `𝖖¹⁶`.  **Depth and cost:** exact to any order.  `Tr 1` and every seed take 1–3 ms through `𝖖⁴⁰` and 15 / 77 ms through `𝖖⁴⁰⁰` (cold cache); finding and certifying the map takes 4 / 19 ms.  In a fresh process, `Tr 1` and every seed through `𝖖⁴⁰` are served through the zoo's own `trace` with no BPS function and no `_generate_bootstrap` run, and neither `bps_kalgebra`, `bps_factor_spectrum`, a bootstrap module nor `trace_uniqueness_proofs` is imported.  The bootstrap stays as the witness `generate` runs.  In `tests/test_ade_serving_guard.py`, zoo pentagon / heptagon pass every check but `geometry` (the zoo's labels are generator indices) and have left the `bootstrap` entry of `EXPECTED_FAILURES`, which keeps `SU3ADKAlg` / zoo a1d4.  Tests: `tests/test_aeven_seeds.py` (28 checks); `tests/test_finite_zoo_traces.py` now exercises the bootstrap's refusal on the pentagon with its route withdrawn.

*Done 2026-09-24 (R19, R20): `SU3ADKAlg` on the curves of `A1DevenKAlg(1)` — geometric labels, and its `T` / `D` seeds from the even-D k = 1 closed forms; no ADE row is served by an orthonormality bootstrap any more.*  `SU3ADKAlg` restricted to SU(2)×U(1) (`su3_to_su2u1_hom`, branching charge `Y`, `3 ↦ 2_{+1} ⊕ 1_{−2}`) is the ungauged `A1DevenKAlg(1)`, whose U(1) fugacity `z` (`E = z⁻¹`) has `Y`-charge 3.  **The map** — a closed form in `su3_ad_kalg` (its "Geometric labels" block): `T_i ↦ {(i, 4), (i + 1, 2)}`, the loop at marked point `i` with the curve from `i + 1` to `i + 3` (magnetic charges ∓1, ±1); `D_i ↦ {(i + 1, 3)}`, the curve from `i + 1` to `i`; a section `T_i^a·D_j^b` to the union.  Each generator carries a `Y`-offset, since `SU3ADKAlg`'s section is canonical (SU(3) is semisimple) and `A1DevenKAlg(1)`'s is fixed by R21's edge-{1, 2} convention: `Δ(T_0) = 0`, `Δ(D_0) = 1`, and ρ gives the rest, `Δ(ρX) = −Δ(X) − 3d`, `d` the power of `E` that ρ produces on the image — `0, 3, 3, 0` on `T_0..T_3`, `1, 2, 1, −1` on `D_0..D_3`, additive over a section's letters.  It is the composite of the two certified zoo maps (zoo a1d4 → `SU3ADKAlg`, `a1d4_seeds`; zoo a1d4 → `A1DevenKAlg(1)`, `a1deven_seeds`), the positive control.  **Certificate:** every ρ-equivariant bijection of the eight generators (2 orbit pairings × 16 shifts), both signs of the U(1), the two base offsets solved exactly from the uniquely matched terms of the 64 generator products: 56 of the 64 candidates fail on the products' labels, 4 have no integral solution, and the 4 certified — every product term for term through the restriction (injective on class functions), ρ on 8/8 — are the closed form composed with the four powers of ρ, the sign alternating; the served closed form is one of the two with `s = +1` (the other is its ρ²-rotation), the composite of the zoo maps.  Negative controls: `Δ(D_0)` moved by 3 (32/64 products), the offsets without ρ's power of `E` (34/64, ρ 3/8), the sign flipped (28/64), the `z`-charge read as `Y` with no factor 3 (28/64).  On the window `2a + b ≤ 8` the 145 sections go to the 145 balanced non-crossing multisets (all of them), and `A1DevenKAlg(1)`'s product of a section's letters' images is that one label with no power of `z`.  **Geometric labels:** `SU3ADKAlg.geometric_label((tile, a, b, p, q)) = (curves, (p, q))` — the hook name of `U1A1AoddKAlg`, `U1A1DevenConeKAlgebra` (on letters) and `UngaugedKAlgebra` (on labels), `A1DnKAlg`'s `(curves, κ)` layout with the SU(3) weight — injective on the window's 580 labels (four weights), ρ the rotation with the weight conjugated.  **Seeds:** `sl3_su3_traces` serves `Tr_T` and `Tr_D` (both parities) as `[Σ_n z^n·Tr_G(L_F·E^n)]/(𝖖²;𝖖²)²_∞` over the closed forms `u1a1deven_seed_characters.seed_trace` at k = 1 (T: the opposite-charge pair; D: the odd curve), restricted to the SU(3) torus by `Y = 3f + Δ`, SU(3) weights `(m1, m2) = (−w, (Y + w)/2)`, and Weyl-symmetrized on totals only, as before; `Tr_1` stays the Kac–Wakimoto vacuum, which the same sum over Creutzig's gauge tower reproduces through `𝖖⁸⁰` (the route's positive control).  The window `|n| ≤ K + 1` is `UngaugedKAlgebra`'s; through `𝖖⁴⁰` the charges contributing reach `|n| = 9` (T) and 20 (D), and a contribution at the window's edge raises.  The forward pass `SU3ElemTraces` is off the serving path and is the witness: the served seeds equal it through `𝖖⁸⁰` (all three) — which also checks the k = 1 closed forms themselves, measured against the transport at lower depth, through `𝖖⁸⁰` by a route that shares nothing with them but the algebra.  **Cost** (a shared four-core machine; exact integers throughout): the three served seeds take 0.07 s through `𝖖⁴⁰`, 0.9 s through `𝖖⁸⁰`, 3.8 s through `𝖖¹²⁰` and 12 s (141 MB) through `𝖖¹⁶⁰`, where the forward pass took 0.4–0.8 s and 15–23 s (410 MB) beyond its `Tr_1` at `𝖖⁴⁰` / `𝖖⁸⁰` (four runs).  What dominates now is `Tr_1` itself, the Kac–Wakimoto `vacuum_character`, unchanged and shared by both routes: about 2 s at `𝖖⁴⁰` and 35–39 s at `𝖖⁸⁰` — where the same ungauging sum over Creutzig's gauge tower, equal to it through `𝖖⁸⁰`, takes 0.3 s (not switched here: a cheap follow-up).  **Through the map:** on the 25 sections with `2a + b ≤ 3`, `SU3ADKAlg`'s trace, restricted, equals `A1DevenKAlg(1)`'s trace of the image through `𝖖⁸` (`𝖖¹²` with `--slow`), taken on the transport route (`seed_closed_forms=False` on the gauged class, since the same closed forms serve `A1DevenKAlg(1)`'s seeds after #1569).  In a fresh process `Tr 1`, `T_0`, `D_0`, `D_1` and zoo a1d4's `Tr 1` and eight seeds through `𝖖⁴⁰` run with no method of `SU3ElemTraces` and no BPS function, and import no module of the even-D classes and no bootstrap (the control: the same script with the forward pass served is flagged).  In `tests/test_ade_serving_guard.py` the `SU3ADKAlg` row has its geometry check (`_geo_su3ad_curves`: balanced non-crossing, injective, ρ the rotation) and passes all nine checks; its `geometry` entry and the `bootstrap` entry (`SU3ADKAlg`, zoo a1d4) left `EXPECTED_FAILURES`, and the positive control that runs `SU3ElemTraces` now calls it directly.  New public name: none beyond `SU3ADKAlg.geometric_label` (an existing hook name, on a new class); listed for the user's veto with the label layout `(curves, (p, q))` and the choice of the served member of the map's ρ-class.  Tests: `tests/test_su3ad_deven_map.py` (21 checks; `--slow` for `𝖖⁸⁰`). *(Same day, later — the follow-up the paragraph names: `Tr_1` too.  The served `Tr_1` is the identity's image, the empty section with offset 0: Creutzig's gauge tower summed over the gauge charge with the measure restored (`_deven_seed_z((), 0, K)` in `_ClosedFormSeeds`), 0.12 s through `𝖖⁶⁰` and 0.9 s through `𝖖¹⁰⁰`, where the Kac–Wakimoto `vacuum_character` takes 11 s and 110 s; the two are equal through `𝖖¹⁰⁰`, and the character stays as the witness (`𝖖⁶⁰` in the default test, `𝖖¹⁰⁰` with `--slow`).  Zoo a1d4's `Tr(1)` is `SU3ADKAlg`'s branched, so it follows; the fresh-process scan also flags the character's functions now, and its control fires on them; in the serving guard the `tr1` phase (`Tr(1)` through `𝖖⁴⁰` under the profile hook) went from 17 s / 18 s to 6 s / 7 s for `SU3ADKAlg` / zoo a1d4.  While checking what depth the served series must hold — the character had computed `Tr_1` three or four orders beyond the depth asked, the closed forms compute exactly that depth — a latent defect surfaced in `product_trace`, under `trace_word` and `inner_product`: it asked its seeds for a padded guess of the depth, `K + margin + a² + b² + 4`, which is one order short on 8 of the 15,424 monomials of the products of two labels with `a + b ≤ 3` and of three generators (e.g. `(T₀²D₀)·(T₀²D₀)`, monomial `(0, 4, 2)`); from a fresh provider those traces came out wrong at the top order (at `𝖖⁸`, `𝖖¹²`, `𝖖²⁰` alike, and with the forward pass as provider too, so the defect predates the switch), and right once an earlier call had deepened the provider.  `product_trace` now asks for the exact depth — per monomial, `K` minus its lowest `𝖖`-power plus the depth its own reduction reads — and `_single_label_trace_z` raises on seeds too shallow instead of truncating.  Tests: `tests/test_su3ad_deven_map.py`, 23 checks.)*

**`A1DnKAlg` (odd `D_v`).**  Its primitives are the `SU(2)`-symmetrised quantum torus of the `D_v` chamber: it agrees with the zoo on `𝖖`-commuting pairs only, has the wrong ρ (A81) and the flat trace `(𝖖²;𝖖²)^{v−1}χ₀` (C.errata item 10; CONFIRMED at `v = 3, 5, 7`).  No relabelling can repair it.  Recommendation: rebuild it (R14: improve, not remove) as a presentation of `A1DoddConeKAlg((v−3)/2)`, either keeping charge labels through the atom↔charge dictionary or adopting the cone's `(word, κ)` labels — the label frame is the user's decision.
*Done 2026-09-23 (`decisions.md` R18, R19).*  The old class body is in `legacy/a1dn_kalg_chamber_torus.py` (measured equal to `SU2QuantumTorusKAlg(chain(v − 1))` on products, ρ and odd-`v` traces over `{−1,0,1}^v`, `v = 3..6`); `A1DnKAlg(v)` (odd `v`) is rebuilt with labels `(curves, κ)` — simple curves `(x, ℓ)` of the `v`-gon with one interior puncture — and products, ρ and trace through `A1DoddConeKAlg((v−3)/2)` by the curve dictionary (a bijection with `𝖖`-commuting ⟺ not crossing and `χ₁`-carrying ⟺ two crossings, verified at `k = 0..4`).  `Tr 1` equals `vacuum_trace_pe(k)` at `k = 0..3` through `𝖖¹²`, `𝖖¹²`, `𝖖¹⁶`, `𝖖²⁰` (past `𝖖^{4k+6}`, the first order where `k` and `k + 1` differ), and at `v = 3` products (576/576), ρ and traces (the identity, `χ₁` and the single curves through `𝖖¹²`) equal the BPS chart `GBPSKAlgebra`'s.  That every single curve's trace is its `seed_trace_ap` seed checks the curve dictionary only: the delegate's trace of one generator is that seed.  Even `v` is refused at construction.  Tests: `tests/test_a1dn_kalg.py`, `tests/test_a1dn_a1dodd_iso.py`, `tests/test_gbps_kalgebra.py`; the stored `a1dodd_cone_tables.py` was dropped the same day (every `k` from the closed-form builder).
*For a later general-`v` curve ↔ charge map (R18 D5 settled on a test-local table at `v = 3`).*  Build it from the ρ-orbits of the curve atoms, not from the reduced nodes' charges.  Measured 2026-09-23 at `v = 5` on `GBPSKAlgebra(FlavouredQuiver(chain(4), [1,1,1,2]))`: two of the four reduced-node charges at the trivial irrep are not curves — `e₁` and `e₂` are each the single-term product of two 𝖖-commuting curve atoms, i.e. degree-2 monomials — and the ρ-orbits of the four node labels (20 labels) have 50 𝖖-commuting unordered pairs where the 20 curves have 90, so no curve map exists on them.  A table of the 20 curve charges built from the curve geometry passes checks made against the BPS chart alone: GBPS ρ moves it as `x ↦ x + 1`; GBPS product classes (one term / several / `χ₁`-carrying) equal the crossing counts on all 190 pairs; and the 70 cones read off those classes are unimodular and tile the box `[−2, 2]⁴` once.  Products and traces then agree with `A1DnKAlg(5)`.  This was a session probe; it is not in the suite.

**Also found (item 1's report).**  `u1aodd_trace_bootstrap.solve_intermediate` has four certification holes (known seeds truncated at a shared cap; `val` returns 0 for a never-solved `(rep, order)`; `_solve` has no consistency check although its docstring claims one; free orders recorded only for columns present in the bucket); at `k = 2`, `K = 40` it returned "certified" values that differ from the singlet rule from `𝖖³²`–`𝖖³⁶`.  `U1A1AoddKAlg` no longer calls it (7a82dab), but the standalone `U1HexagonKAlg`, `U1OctagonKAlg`, `U1DecagonKAlg`, `U1DodecagonKAlg` still serve their chord seeds from it.  Measured 2026-09-23: the standalone and general presentations differ at single chords by the sign of the E-power — `Tr_standalone(L·E^e) = Tr_{U1A1AoddKAlg}(L·E^{−e})` on every single-chord label tried through `𝖖¹²` (`U1OctagonKAlg` 20/20, `U1DecagonKAlg` 40/40; unflipped only the `e = 0` labels agree, 4/20 and 8/40).  The named ungauged classes (`OctagonKAlg`, `DecagonKAlg`, `DodecagonKAlg`) are built by `ungauge_u1polygon(k)` on these standalone classes, so they inherit `solve_intermediate`'s deep errors, while `ungauge_u1a1aodd(k)` does not.  Serving the standalone chord seeds through the singlet rule (via `e ↦ −e`), or rebuilding the named classes on `U1A1AoddKAlg(k)`, is a proposal for the user (the named classes' labels would change in the second case).
*Done 2026-09-23 (`decisions.md` R19: both halves, under the mandate).*  `U1HexagonKAlg` keeps its labels and is served by `U1A1AoddKAlg(1)` through `(F, e) ↦ (F, −e)` — products, `ρ`, `ρ⁻¹`, traces and the chord seeds of `_trace_residual` (121/121 generator products against its frozen table, `ρ` 11/11, the 21 chord seeds equal to the bootstrap's through `K = 40`; without the flip 48/121 products and 18/21 seeds differ).  The stand-alone `U1OctagonKAlg` / `U1DecagonKAlg` / `U1DodecagonKAlg` are in `legacy/` with their tables; the named `OctagonKAlg` / `DecagonKAlg` / `DodecagonKAlg` are `UngaugedPolygonKAlg(k)`, the ungauging of `U1A1AoddKAlg(k)`, with the labels of `U1A1AoddKAlg(k)` (balanced multisets of diagonals, R20) and `mult_generators()` 24 / 65 / 144.  The label change is the same-diagonal dictionary, certified on every generator product and on `ρ`, `ρ⁻¹` (676 / 1369 / 3136) and carrying the recorded stand-alone traces through `𝖖¹²`; the measured sign above is its `k = 2, 3` form — at `k = 4` the stand-alone `E` was already the diagonal-geometric one (`E ↦ E`, with the letter index of types 2, 3 shifted by 9 and of type 4 by 7).  The rebuilt classes read the ungauging with `μ ↦ μ⁻¹`, so that `Tr((F, e)) = μ^{e}·Tr((F, 0))` — the section convention their lift states (the old trace contradicted it), the orientation of `octagon_objects` and of the zoo; under the dictionary the `μ`-graded traces are then unchanged at `k = 2, 3` and conjugated at `k = 4`.  `ungauge_u1polygon` is retired with the stand-alone classes; `solve_intermediate` is on no serving path (`repo_audit.md` C.errata item 18).

*The flavour lift of the ungauged classes (2026-09-23, later the same day).*  Every ungauged class's `r_label_decompose` now agrees with its own trace.  One rule on `UngaugedKAlgebra`: section `(F, 0)`, U(1) weight minus the E-power — its trace gives `Tr((F, e)) = z^{−e}·Tr((F, 0))`, so `E` is `z⁻¹` — with the gauged class's own irrep beside it.  Per class: `ungauge_u1a1aodd(k)` and `ungauge_u1a1deven(k)` key `(−e,)` / `(b, (−e,))` (was: the label as its own section, key `(0,)`); `A1DevenKAlg(k)` `(b, (−e,))`, `b` the SU(2) unit (was the same defect); `HexagonKAlg` `(−e,)` by delegation (was `(e,)`, which its trace contradicts; the pin in `tests/test_hexagon_kalg.py` changed); `UngaugedPolygonKAlg(k)` and the named k ≥ 2 classes `(e,)`, the rule read through their `μ ↦ μ⁻¹`, unchanged.  On 20 random labels per class, at k = 1..3 for both families, the trace is R-linear for the lift, `to_R_form` round-trips, the key is a single irrep, `E^e·L_{(F,0)}` is `χ_key·L_{(F,0)}`, and the opposite sign fails; before the change the same probe passed the rebuilt `UngaugedPolygonKAlg` and failed `HexagonKAlg` (2/20).  Numbers and tests: `repo_audit.md` C.errata item 18.

## M. [A₁,E₇]: every seed and the vacuum in closed form (2026-09-23)

Ruling R17's route 3, verified and served.

**The closed form** (`finite_kalgebras/e7_seeds.py`).  Every seed of `FiniteE7KAlgebra` is

    T_i = N_i / ((q;q)_∞ θ(μ)),    θ(μ) = Σ_n 𝖖^{n²} μⁿ,    (q;q)_∞ = Π_{n≥1} (1 − 𝖖^{2n}),

with `N_i` a short integer combination of monomials `𝖖^s μ^t` times the ten products `th·ψ_aψ_b` (`0 ≤ a < b ≤ 4`) of `ψ_a = Σ_n 𝖖^{5n²+(2a−4)n} μⁿ` and `th2 = Π_{m≥0}(1−𝖖^{10m+2})(1−𝖖^{10m+8})` or `th4 = Π_{m≥0}(1−𝖖^{10m+4})(1−𝖖^{10m+6})`.  Nine recipes, three to twenty-four terms (after merging equal monomials), one per ρ-orbit representative (`0, 1, 2, 3, 5, 7, 12, 13, 21`; seeds 0 and 1 have one recipe); the other 81 seeds follow by ρ-equivariance of the trace in its element form on this cone tier, `T_{ρ(i)}(μ) = μ^{−δ_i} T_i(μ⁻¹)`, from the class's own `E7_RHO_PERM` / `E7_RHO_DELTA` (nine orbits of length 10 cover the 90 seeds).  `Tr 1` is the product

    Π_{n≥1} (1−𝖖^{10n})²(1−𝖖^{10n−2})(1−𝖖^{10n+2})(1+μ^{±1}𝖖^{10n−1})(1+μ^{±1}𝖖^{10n+1})
      / Π_{n≥1} (1−𝖖^{2n})(1−𝖖^{2n+2})(1+μ^{±1}𝖖^{2n+1}),

the Euler–Poincaré character of the minimal reduction of the Kac–Wakimoto boundary-level `sl₃` character at `k = −3 + 3/5` (the Bershadsky–Polyakov vacuum at level `−12/5`; a literature prior, used here only as a measured identity).  Cost: all 90 seeds in 0.5 s at `𝖖⁴⁰` and 17 s at `𝖖¹²⁰`; `Tr 1` in 0.2 s at `𝖖¹²⁰`.

**How it was found and what certifies it.**  The research pass decomposed the seeds by U(1) charge: after multiplying by `(q;q)_∞ θ(μ)`, every charge sector of each product `th·ψ_aψ_b` is a monomial times a Virasoro M(3,5) character (measured through `q¹²⁰`), and each numerator is a sparse integer combination of the ten products, selected on the route-2 bootstrap's values through `𝖖¹¹`–`𝖖¹⁴` and validated on the deeper orders.  The recipes were then re-implemented from their text alone and compared (`tests/test_e7_seeds.py`, 12 checks):

| witness | range | result |
|---|---|---|
| route 2, the certified orthonormality bootstrap (`generate_u1`, ρ-fold, under the recorded hypothesis `\|μ\| ≤ 4k`; 569,185 rows, 17,777 determined values) | through `𝖖²⁰` | 90/90 seeds equal |
| route 2 with the ρ²-FOLD (does not use the ρ-rule, so it checks the 81 transported seeds independently) | through `𝖖⁶` | 90/90 seeds equal |
| route 2 with `Tr 1` supplied by the Bershadsky–Polyakov product, solved to `𝖖²⁴` (645,481 rows, consistent) — a JOINT test of the vacuum and the seeds against the axioms' rows | through `𝖖²²` | 90/90 seeds equal, including the held-out orders `𝖖²¹`–`𝖖²²` (980 coefficients) |
| the Nahm sum on the BPS spectrum (`elem_traces._vacuum_rps`), for `Tr 1` | through `𝖖²⁰` | equal (77 terms) |
| series-engine controls: the Jacobi triple product for `θ(μ)`; `D·D⁻¹ = 1` | through `𝖖⁴⁰` | equal |
| negative controls: one extra term deep in a recipe; the levels `u = 4, 7` in the vacuum | — | caught (the vacuum from `𝖖⁶`, `𝖖⁸`) |

Consistency checks: the ρ-rule closes around every orbit, and every closed form obeys the bootstrap's `μ`-support hypothesis through `𝖖⁶⁰`.  The q-orders `𝖖¹⁵`–`𝖖²⁰` were held out from the fit (R17's acceptance condition), and `𝖖²¹`–`𝖖²²` were never seen by it.  The deeper run takes its vacuum from the product because route 2 with the Nahm vacuum exceeded 9.2 GB computing that vacuum at `𝖖²⁶`; a wrong vacuum at those orders would have shown as an inconsistency or as different seeds.

**After.**  `FiniteE7KAlgebra.trace` serves these closed forms (`elem_traces._seed_series`, beside the W₃ entry): `Tr 1` through `𝖖³⁰` in 0.01 s and all 90 seeds through `𝖖¹⁶` in 0.08 s, with neither the BPS engine nor the bootstrap loaded.  Leading multiplicativity holds (`val Tr(g^m) = m·val Tr(g)` for `g = L_0, L_5, L_12`), and orthonormality holds in the element form — `Tr(ρ(L_a)·L_b) = δ + O(𝖖)` with `ρ = rho_element`, on all pairs of a 10-label window that includes the generators with `δ = −1, −2`.  The label-form `inner_product` is off by the unit character `μ^{δ(a)}` on those generators: the encoding split between the label `rho` and `rho_element` recorded for this tier in `repo_audit.md` A74, not a trace defect.

**Standing.**  MEASURED closed forms, not derived.  The reading of the ten products as Bershadsky–Polyakov module characters is a literature hypothesis, confirmed in the repo only for the vacuum; nothing served depends on it.  Route 2 stays in the repo as the witness.

*The u(1)-gauged `[A₁,E₇]` served from these closed forms (2026-09-23, later the same day).*  `U1E7ConeKAlgebra` now traces every magnetically neutral label, the E-tower included, through `FiniteE7KAlgebra`: `Tr(L_ℓ) = [μ^{−r}]((𝖖²;𝖖²)_∞²·Tr_{[A₁,E₇]}(L_z; μ))` for `φ(L_ℓ) = μ^r·L_z`, the gauging of the U(1) flavour; magnetic labels stay exact zeros.  The label map `φ` (`E ↦ μ`) sends 90 labels of the neutral sector to the zoo's 90 generators.  70 of them are the cone presentation's neutral chord atoms, forming 7 of the zoo's 9 ρ-orbits.  The other 20 fill the two remaining orbits: each is a product of a `c0 = +1` and a `c0 = −1` atom, a generator of the neutral sector that the gauged cone presentation does not list as an atom.  So the zoo's count of 72 generators with last lattice coordinate 0 is not the neutral count: every zoo generator lies in the magnetically neutral sector.  That coordinate is the component along E₇'s branch node, and it is not the preimage's E-power (measured: the two agree on 40 of the 90).  **Finding (the flow as a finder only).**  The flow's flavour-refined traces of the 70 atoms, matched against the 90 closed-form seeds (μ-shifts and `μ → μ^{−1}` allowed), leave 10–40 candidates per atom and both orientations.  ρ-equivariance and the q-commutation graph cut this to `E ↦ μ` with exactly five solutions, the ρ²-translates of one (ρ² moves no trace).  The 20 products of a `c0 = ±1` atom pair are then read off generator products.  **Certificate, by products** (`tests/test_u1e7_cone_kalgebra.py`):
- all 8100 ordered generator products agree under `φ`, in 2.4 s;
- every one of the 2600 zoo 6-cones gives a single-term image product with the zoo's phase;
- 400/400 random light products agree;
- ρ commutes with `φ` on 90/90 generators and on 200/200 random labels;
- four perturbed maps fail within 2–10 products: a swap inside a ρ²-orbit, one μ-shift off by one, `E ↦ μ⁻¹`, and one orbit translated by ρ².

`φ⁻¹` is linear on each zoo cone, and the class inverts it by an exact chord-multiset cover, unique on 500/500 random neutral labels.  **Acceptance:**
- the E-tower equals the gauged Nahm sum (its witness) through `𝖖¹⁰`;
- the traces equal the flow's on 16 held-out labels (13 through `𝖖⁸`, 10 of them nonzero; 3 through `𝖖⁶`, all three zero), and the `[μ^{+r}]` reading fails;
- ρ²-twisted cyclicity holds on 41/41 pairs, including the pair recorded as failing under the former bootstrap;
- `⟨a,b⟩ = δ + O(𝖖)` on 400/400 pairs of neutral and magnetic atoms, in label form: `E` sits in the label and the coefficient ring is `Z`, so the A74 split does not arise;
- a neutral trace at `K = 20` loads no flow, BPS engine, bootstrap or Nahm engine.

The E-tower's vacuum was moved from the Nahm sum to this closed form: the Nahm sum costs 30 s at `𝖖¹⁴` and grows about threefold every two orders, and `trace_element` widens `K` to `𝖖¹⁸` in a ρ²-cyclicity check of a `c0 = ±3` pair.  **Open:** a deep composite zoo label is slow through the zoo's own Layer-1 reduction.  `((21,2),(47,2),(62,2))` took 409 s under a profiler, over 212,669 distinct subwords, and the zoo's own `multiply` exhausted a 6 GB address space on a random product of cone monomials with up to three generators at powers ≤ 2.  Labels of the depth met in the tests trace in under a second.  Record: `gauged_qtcone_from_rg.md`, `repo_audit.md` C.errata item 20.


## N. The seeds maps as `KAlgebraIso`s (2026-09-26)

Ruling R22 (user, 2026-09-26) records the question *"Are the seeds
KALgebraIso's?"* (verbatim).  They were not: each seeds module held a
generator map certified on every generator product, on ρ and on traces, and
`finite_kalgebras.objects.kalgebra_object` did not hold the family classes.
A `KAlgebraIso` multiplies coefficients through and compares traces in one
ring, so each iso now runs from the zoo's Z-form wrapper (flavour moved from
the coefficients into the labels) onto the family class.  The maps and their
flavour normalisations are the served ones; the details are in the three
modules' docstrings ("The KAlgebraIso").

| entries | source: the zoo's Z-form wrapper | target | constructor |
|---|---|---|---|
| `a3`, `a5`, `a7` (and the aliases) | `FiniteU1ZKAlgebra(native)`, base-changed along `μ ↦ z⁻¹` (one `R(U(1))` in two coordinates) | `ungauge_u1a1aodd(k)` | `aodd_seeds.kalgebra_iso` |
| `a1d3`, `a1d5`, `a1d7` | `FiniteSU2ZKAlgebra(native)` (new; `FiniteA1D3ZKAlgebra` is its a1d3 instance) | `A1DoddConeKAlg(k)` | `a1dodd_seeds.kalgebra_iso` |
| `a1d4`, `a1d6`, `a1d8` | `FiniteSU2U1ZKAlgebra(native)` (same ring, `s = +1`) | `A1DevenKAlg(k)` | `a1deven_seeds.kalgebra_iso` |

The u1 and su2u1 wrappers gained the flavour-lift coordinate
(`r_label_decompose` / `r_label_compose`) for the section check.
**A defect found by the battery, and fixed:** `FiniteSU2U1ZKAlgebra.trace`
dropped the SU(2) character of its label (`Tr(χ₁)` returned `Tr(1)`).  The
trace check failed on `χ₁` at a1d4 and a1d6 until the fix.  The wrapper is on
no serving path.

**Certificates** (`tests/test_{aodd,a1dodd,a1deven}_seeds.py`; measured
2026-09-26 on a shared four-core machine).  Samples: the identity, every
generator and the flavour characters.  All five checks of `verify_all` pass:
unit, round trip, multiplicative on every ordered pair of the non-identity
samples each way, ρ-equivariant, trace-equivariant through `𝖖¹²`.
`verify_maps_section_to_section_1drep` holds both ways, and `inverse∘map = id`
holds on every label of the products:

| entry | samples | pairs each way | time | round trip on product labels |
|---|---|---|---|---|
| a3 / a5 / a7 | 9 / 27 / 68 | 64 / 676 / 4,489 | 0.2 / 1.0 / 8.9 s | 36+40 / 246+301 / 1,343+1,954 |
| a1d3 / a1d5 / a1d7 | 8 / 22 / 44 | 49 / 441 / 1,849 | 0.1 / 0.2 / 2.9 s | 27+27 / 163+163 / 619+619 |
| a1d4 / a1d6 | 12 / 43 | 121 / 1,764 | 0.3 / 3.3 s | 77+76 / 871+819 |
| a1d8, `--slow` | 124 | 15,129 | 242 s | 6,512+6,099 |
| a1d8, default | 124 | 1,089 (every 4th generator and the flavour characters) | 37 s | 1,295+1,388 |

The trace check compares independent routes at a1d4 (`SU3ADKAlg` serves the
zoo) and at a1d5 / a1d7 (the zoo's `a1d5_layer2` / `a1d7_layer2`).  Elsewhere
the zoo's seeds are served through the same map.  At a1d3 the maps also pass
from the existing `FiniteA1D3ZKAlgebra()`.  Negative controls, each failing
the battery: a flipped flavour normalisation (`μ ↦ z` at a3 and a5; `s → −s`
at a1d4 and a1d6); one generator's offset moved by 1 (a1d4, a1d6); two
generators transposed (a3, a5, a1d4, a1d6); two ρ-orbits swapped, or one
shifted alone (a1d5, a1d7).

**Object layer.**  For the twelve flavoured ids, `kalgebra_object` adds a
second component: 'z-form' (the wrapper of the registered 'cone-frozen'
instance) and 'ungauged-u1a1aodd' / 'a1dodd' / 'a1deven' (the family
instance the seeds serve from), with the iso as witness.  No witness joins it
to {'cone-frozen', 'bps'}: a `KAlgebraIso` cannot move a coefficient's `μ`
into a label.  `iso` across the components raises `KeyError`; the link is the
wrapper holding the 'cone-frozen' instance.  `preferred` is unchanged for the
existing tags (`tests/test_kalgebra_object.py`).  **`[A₁,A₂ₖ]`:** the object
layer's 'cone-frozen' → 'a1a2k' witness (`match_generators`) is `aeven_seeds`'
served map itself, ρ⁰, at the pentagon and the heptagon (no new
construction).

## O. [A₁,E₈]: Layer 1 on canonical labels — the deep-label wall is gone (2026-09-26)

The user, 2026-09-26: *"Maybe your newly gained experience can crack E_8"*.
After section K the seeds of `FiniteE8KAlgebra` were exact to any order; what
remained was Layer 1, the ρ²-cyclicity reduction of a label to the seeds.

**The wall, measured on main** (a shared four-core machine, each label in its
own process under a 6 GB cap).  Labels of degree ≤ 4 traced in ≤ 3 s at
`K = 4` (60/60: the recorded "degree-4 wall" was stale).  At `K = 8`, degree 12
was the limit: a³b³ up to 82 s and 4.8 GB, a³b³c³d³ 78 s and 4.7 GB, the full
cone (degree 8) 40 s and 2.3 GB, and the full cone squared (degree 16) out of
memory, 3 of 3.  Products stayed cheap (40/40 in ≤ 0.1 s).  **Cause:** the word
route memoises the seed expansion of every sub-word it meets.  On a³b³
`[[2,3],[51,3]]` that is 267,611 words, about half of them spanning several
cones, holding 52.3 M polynomial terms, which is where the 4.8 GB went.

**The route** (`ConeData.layer1_on_labels`, `_simplify_trace_on_labels`; set
on `_ExportedE8ConeData` by its generator).  The same two identities as the
word route — ρ²-twisted cyclicity on one generator (Forms B and A) and
`Tr∘ρ² = Tr` — but every intermediate is a canonical label: for `L = L_g·L'`
the tag `ρ^{∓2k}(g)` moves on while it q-commutes with `L'` (the word route's
cocycle swap), and at the first collision the class's own `multiply` expands
`L'·L_h` into canonical labels.  Every letter and both forms are tried and the
expansion whose largest term has the least total degree is taken — the best of
eight rules measured on five heavy labels (the next best stored 2–6× more
labels; all eight gave the same values).  Labels are keyed by their
ρ²-orbit representative only: folding by ρ would build axiom 5 (ρ-equivariance
of the trace) into the route and make its verifier a tautology on the zoo,
where `kalgebra.md` lists it as genuine content.  Two passes: the reduction
graph first, then the seed expansions bottom-up, each freed after its last
parent has used it.  The seeds, and Layer 2, are the word route's.

**Positive controls** (`tests/test_cone_layer1_labels.py`, and the scratch
measurements of 2026-09-26):

| check | result |
|---|---|
| E₈, the 60 labels of degree ≤ 4, `trace` both routes at `K = 8` | 60/60 equal (33 nonzero) |
| E₈, 42 deep labels (degree 5–16) against the word route's recorded `K = 8` values | 39/39 equal (the 3 of degree 16 have no word-route value) |
| E₈, the 39 deep labels the word route reaches, both routes at `K = 30` | 39/39 equal (33 nonzero) |
| E₆, 56 random labels of degree 2–10, both routes at `K = 12` | 56/56 equal (32 nonzero) |
| E₇ (flavoured: the ρ^{∓2} steps carry `μ`-units), 36 random labels of degree 2–8, both routes at `K = 10` | 36/36 equal (22 nonzero); at degree 10 the word route ran out of memory (6 GB) on the first label, which the labels route had traced |
| pentagon, heptagon, E₆, E₈ in the test, `K = 12` | equal |
| negative control: `U1A1AoddKAlg(1)`, whose ρ⁻² sends the wrap chord `(1,0)` to three letters | refuses, naming the word route |

The reduction can be steered, so the check that means most is the one against
a different derivation: all eight rules gave equal values on the five heavy
labels.

**Reach** (`K = 30`; the derivation does not depend on `K`):

| label | word route | labels route |
|---|---|---|
| a³b³ `[[2,3],[51,3]]` (degree 6) | 64 s, 4.8 GB | 2 s |
| a³b³c³d³ (degree 12), 3 labels | 10–76 s, up to 4.7 GB | 1–4 s, ≤ 0.3 GB |
| full cone squared (degree 16), 3 labels | out of memory at 6 GB | about 8, 19 and 67 s; 0.5 GB for the last, alone in its process |
| full cone cubed (degree 24) | — | 274 s, 0.74 GB |

**The axioms on deep labels** (`K = 30`, the 42 deep labels above, the three of
degree 16 included): `Tr L = O(𝖖)` with no negative powers, and ρ-equivariance
of the trace, `Tr(ρL) = Tr(L) = Tr(ρ⁻¹L)` — the route folds only by ρ², so these
compare different derivations: 42/42 (33 nonzero).

**Aside: the seed expansion is not unique, the character expansion is.**  The
seventeen seed traces span only five W₃(3,8) characters, so a label's seed
expansion depends on the derivation (51 of the 60 small labels expand
differently in the two routes, with equal traces).  The five characters are
linearly independent over `ℤ[q_paper^±]` for coefficients of degree ≤ 110
(rank test mod `2⁶¹−1`), so a trace's expansion in characters is unique — and
large: `Tr(a³b³)` needs coefficient polynomials from `𝖖⁻¹⁸` to `𝖖¹⁸⁰`, with
coefficients near 9·10⁵, for the series `𝖖⁶ + 𝖖¹⁰ + 𝖖¹² + 2𝖖¹⁴ + ⋯`.  Hence
the route keeps seed leaves: characters would save a factor below 2.

**Standing.**  E₈'s traces are exact at any order on every label the route
reaches, and it reaches degree 24 in minutes; the cost grows with degree
(about 80 k labels in the derivation at degree 24, measured with the route's prototype), not with `K`.  E₈ still has
no companion subsection (R13); one is a proposal for the user.  E₆ keeps the
word route (see "The rest of the zoo" below).

**E₇ switched the same day** (`FiniteE7KAlgebra`, through
`generate_finite_kalg`'s new `layer1_on_labels` option, set for `e7` only).
E₇'s `_canonical_rho2_orbit_rep` declines to fold (its label-level ρ² differs
from the element-level one by `μ`-shifts), so the route folds its labels
itself, element-level: `ρ²(L_w) = μ^{δ(ρw)−δ(w)}·L_{ρ²w}` (the unit of
`_rho2_twist_unit`), hence `Tr L_w = μ^{δ(ρw)−δ(w)}·Tr L_{ρ²w}`, and an orbit
that closes on a unit `≠ 1` proves `Tr L_w = 0`.  Without that folding the route
kept all 91 seeds apart and the expansions grew heavy: on `((21,3),(47,3),(62,3))`
451 s and 2.4 GB (root expansion: 91 seeds, 25.8 k powers of `𝖖`), with it
135 s and 0.8 GB (19 seeds, 5.6 k).  With it, besides the 36/36 control above
(re-run after the folding: 36/36):

| label (`K`) | word route | labels route |
|---|---|---|
| `((21,2),(47,2),(62,2))`, the companion's (`𝖖¹²`) | 147 s, 5.2 GB (159 s and 5.4 GB at `𝖖⁴`, as `comp:e7` recorded) | 11 s, 0.25 GB; both `μ⁻²𝖖¹² + O(𝖖¹³)` |
| `((6,10),)` (`𝖖¹⁰`) | out of memory at 6 GB | 9 s, 0.16 GB (`𝖖¹⁰ + O(𝖖¹¹)`) |
| `((6,16),)` (`𝖖¹⁰`) | — | 194 s, 0.9 GB |
| three labels of degree 12, one of degree 16, up to four distinct generators (`𝖖¹⁰`) | — | 48–870 s, 0.45–3.3 GB; `((21,4),(47,4),(62,4))` 870 s and 3.2 GB (out of memory at 6 GB before the folding) |

The `comp:e7` battery rows pass on the switched class (34 checks), the ρ²-orbit
bootstrap of `comp:e7/traces` included; the companion's paragraph "How far the
traces are defined" (`comp:e7`) now quotes these.

**The rest of the zoo (2026-09-26).**  The route applies to every zoo entry:
each generator's ρ^{±2} image is one generator in all twelve classes
(`hexagon`, `octagon` and `decagon` in `FINITE_KALGEBRAS` are the A₃, A₅ and A₇
classes again).  Positive control, 20 random labels of degree 2–6 per class,
both routes at `K = 8`: 240/240 equal (162 nonzero).  Cost at `K = 8`, one
process per class, 150 s cap per trace at degrees 8–10 and 300 s at 12, 4.5 GB
address space at 12:

| class | labels route vs word route |
|---|---|
| A₅ | degree 10: 13 s vs 19 s; degree 12: 9.4 s vs 29.5 s, 2.1 s vs 5.7 s |
| A₇ | degree 10: 30 s vs 42 s; degree 12: 59 s vs out of memory, 3.0 s vs 11.0 s |
| A₁D₆ | degree 10: 73 s vs over 150 s (2.1 GB), 4.7 s vs 6.2 s |
| A₁D₈ | degree 8: 56 s vs over 150 s (4.2 GB), 23 s vs out of memory; degree 10: over 300 s vs out of memory, on both labels (the first completes in 874 s, below) |
| A₁D₄, before the Layer-2 rewrite below | degree 8: 92 s vs 61 s on `((6,8),)`; degree 12: over 300 s vs 212 s on `((6,6),(7,6))`, 17.7 s vs 30.5 s |
| A₁D₄, after it | degree 8: 11.3 s vs 15.1 s on `((6,8),)`, 1.7 s vs 2.5 s; degree 10: 0.6 s vs 0.8 s, 5.9 s vs 8.8 s; degree 12: 38.8 s vs 52.0 s on `((6,6),(7,6))`, 17.9 s vs 31.1 s (controls re-run: 20/20) |
| E₆ | degree 10: 0.7 s vs 6.8 s on one label, equal on the other; degree 12: equal (≤ 1.4 s) |
| A₁D₇ | equal at degree 12 (9–10 s) |
| pentagon, heptagon, A₃, A₁D₃, A₁D₅ | ≤ 1.2 s either way through degree 10 |

Switched (`generate_finite_kalg`'s `_LAYER1_ON_LABELS`, now A₅, A₇, A₁D₄, A₁D₆,
A₁D₈ and E₇): the classes that gain.  The test
(`tests/test_cone_layer1_labels.py`) checks 8 labels of each at `K = 8`, and 12
at `K = 10` with `--slow`.  Kept on the word route: E₆, whose seeds come from
the orthonormality bootstrap, whose equations are Layer-1 reductions
(switching would change that system, for no gain at degree 12); A₁D₇ and the
small classes, which gain nothing.

**Layer 2 at high order (A₁D₄).**  A deep label's seed expansion carries large
cancellations: on `((6,6),(7,6))` the coefficients run from `𝖖⁻⁸³` to `𝖖⁸³`
(from `𝖖^{∓37}` at `((6,4),(7,4))`; the word route's expansion spans the
same), so a trace through `𝖖⁸` needs every seed through about `𝖖⁹¹`.  Layer 1
took 2.6 s there; serving the seeds took the rest — 393 s of a 420 s profile in
`sl3_su3_traces.sym_to_char`, whose highest-weight peel ranked weights by a
`Fraction` inner product (3.4 M calls), then the branching to SU(2)×U(1).
Three exact rewrites, each checked against the code it replaces:

* `sl3_su3_traces.sym_to_char` reads the coefficient of `χ_λ` off `z·Δ` at
  `λ+ρ` (the Weyl character formula; `Δ` the Weyl denominator), and
  `antisym_to_char` reads an alternant combination's strictly dominant
  coefficients; both check the Weyl (anti)symmetry they rely on.  Identical
  outputs on 300 random virtual characters, the 74 z-Laurents that SU3AD's
  traces through `𝖖³⁰` feed in (up to 721 weights), the 31 numerators of
  `vacuum_character(30)` and 300 random alternants; old and new both raise on
  three asymmetric inputs.
* `zplus_ring.su3_to_su2u1_hom` branches by Gelfand–Tsetlin interlacing: as a
  U(3) irrep `χ_(p,q)` is `(p+q, q, 0)`, which restricts to the sum of
  `(μ₁, μ₂)`, `p+q ≥ μ₁ ≥ q ≥ μ₂ ≥ 0`, each once — SU(2) label `μ₁−μ₂` at U(1)
  charge `3(μ₁+μ₂) − 2(p+2q)`.  The weight-system route it replaced stays as
  the witness (`_su3_to_su2u1_by_weights`): equal on all 496 irreps with
  `p+q ≤ 30` (`tests/test_su3_to_su2u1_hom.py`).
* `zplus_ring.RingHom.apply_RElement` accumulates into one dictionary (it
  rebuilt the partial sum at every term).

What remains on that label is the final product of the seed series with the
coefficients over `R(SU(2)×U(1))` (78 of 101 s in a profile).

**A₁D₈ at degree 10.**  On `((26,3),(70,2),(115,5))` the labels route
finishes Layer 1 in 570 s (16,610 labels planned, 31 seeds, 1.55 GB; the 300 s
caps above cut it short), with coefficients from `𝖖⁻¹²²` to `𝖖¹³¹`; Layer 2
then takes 304 s, serving the seeds through about `𝖖¹³⁰`, and the trace through
`𝖖⁸` is `0 + O(𝖖⁹)` — about 15 minutes in all, where the word route runs out of
memory.  The time is spread over the planning's cone multiplications (two per
candidate, about seven candidates per label).  Exact memos for `cross_product`
and `canonicalize_cone_label` gave bit-identical reductions on six labels across
A₁D₈, A₁D₆, A₁D₄, A₇, E₆ and E₈ but saved only 7 % (A₁D₈ degree 8: 39.4 s →
36.8 s), so they were not kept.  Half of the multiplications only read off the
`±𝖖^e` of `L_g·L_rest` inside one cone — the cheap half, as those products never
collide; that equals `𝖖^{±Σ_x p_x c(g,x)}` on all 9,970 sampled products of the
twelve other classes, but in A₇ and A₁D₈ 200 of 1,960 land on another
decomposition of the same lattice point (the class's `canonicalize_cone_label`
rewrites it), so skipping them needs that canonicalisation.  Not done.
