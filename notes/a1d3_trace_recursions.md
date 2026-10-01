# The trace of `A_𝖖([A_1, D_3])` — recursions, cyclicity, and determination from `O(𝖖)`

The `[A_1, D_3]` counterpart of `pentagon_kq_proof.md` §7, written to the
same standard: **explicit recursion relations** reducing every trace to
the three elementary traces, an **elementary demonstration of
`ρ²`-cyclicity**, and the **determination of the trace from the `O(𝖖)`
axioms** up to overall scale.

Certificate: `scripts/verify_a1d3_trace_recursions.py` (exact
arithmetic, 24 checks). Implementation: `implementations/a1d3_kalg.py`.
Companions: `pentagon_kq_proof.md` (the model), `a1a2k_kq_proof.md`
(the `[A_1,A_{2k}]` family), `finite_type_kalgebras.md` §5 (the
uniqueness certificates).

---

## 0. Scope — what this document does and does not establish

**Establishes.** Given that the relations below define an associative
algebra and that `Tr` is a `Z[𝖖,𝖖⁻¹]`-linear functional satisfying
`ρ²`-twisted cyclicity `Tr(xy) = Tr(ρ²(y)x)`:

* every trace of a canonical basis element is a `Z[𝖖,𝖖⁻¹]⊗R_{SU(2)}`
  combination of `Tr 1`, `Tr T_0`, `Tr D_0` (§3–§4);
* cyclicity forces **no relation** among those three, so the space of
  such functionals is **exactly** three-dimensional (§5);
* the `O(𝖖)` axioms fix `Tr T_0/Tr 1` and `Tr D_0/Tr 1` uniquely (§6).

**Presupposes, and does not prove here.** That the relations *do* define
an associative algebra (existence), and that the cyclicity axiom is at
`ρ²` specifically. Both are load-bearing and neither is a formal
consequence of the reduction: an adversarial check found that negating
the `𝖖`-exponents on the six `T·D` / `D·T` relations leaves every
statement of §4 intact while destroying associativity (66 of 216
generator triples fail), and that replacing `ρ²` by `ρ` in the cycling
step likewise leaves §4 intact while changing the answers on 1915 of
1948 words. So §4 alone characterises a family of rewriting systems, and
it is §1's relation table plus the `ρ²` in §5 that single out this
algebra. See `RESEARCH_GOALS.md` Goal 2.11 for the separate, still-open
"minor miracle", which this document does **not** address.

---

## 1. The data

Six multiplicative generators `T_i`, `D_i`, `i ∈ Z/3`, with

```
T_i T_{i+1}  = 1 + 𝖖⁻¹ χ₁ D_i + 𝖖⁻² D_i²      T_{i+1} T_i = 1 + 𝖖 χ₁ D_i + 𝖖² D_i²
D_i D_{i+1}  = 1 + 𝖖⁻¹ T_{i+1}                D_{i+1} D_i = 1 + 𝖖 T_{i+1}
T_i D_{i+1}  = χ₁ + 𝖖⁻¹ D_i + 𝖖 D_{i-1}       D_{i+1} T_i = χ₁ + 𝖖 D_i + 𝖖⁻¹ D_{i-1}
T_i D_i      = 𝖖⁻² D_i T_i                    T_i D_{i-1} = 𝖖² D_{i-1} T_i
```

`χ₁` the `SU(2)` fundamental character; `ρ(T_i) = T_{i+1}`,
`ρ(D_i) = D_{i+1}` fixing every `χ_k`, so `ρ³ = 1` and
`ρ²(T_i) = T_{i-1}`, `ρ²(D_i) = D_{i-1}`.

The canonical basis is `𝖖^{ab} T_i^a D_i^b` and `𝖖^{-ab} T_i^a D_{i-1}^b`,
dressed by `χ_k`. So there are **six cones**: three of *type I*,
`(T_i, D_i)`, and three of *type II*, `(T_i, D_{i-1})` — and within a
cone the two letters `𝖖`-commute, by the last line above.

*(Certificate C1: the implemented `_interaction` / `_q_commute_twist`
tables agree with a from-scratch transcription of these eight lines on
all 36 ordered letter pairs.)*

## 2. Two reductions before any work

**2.1 `χ` is a spectator.** `Tr` is `R_{SU(2)}`-linear, so
`Tr(χ_k · x) = χ_k · Tr(x)`; the flavour dressing never has to be
carried through a recursion. *(Certificate C2.)*

**2.2 The trace is `Z_3`-invariant, so one cone of each type suffices.**
Cyclicity at `y = 1` gives `Tr ∘ ρ² = Tr`, and `ρ²` generates `Z_3`, so
`Tr ∘ ρ = Tr`. Hence `Tr(T_i^a D_i^b)` and `Tr(T_i^a D_{i-1}^b)` are
independent of `i`. Fix `i = 0` and write

```
U_{a,b} := Tr(T_0^a D_0^b)        (type I)
V_{a,b} := Tr(T_0^a D_2^b)        (type II, D_2 = D_{0-1})
```

Both are traces of *literal monomials*; the canonical basis element
carries an extra `𝖖^{±ab}`, which is a scalar. Note `U_{a,0} = V_{a,0}`.
The three elementary traces are `U_{0,0} = Tr 1`, `U_{1,0} = Tr T_0`,
`U_{0,1} = Tr D_0`. *(Certificate C2: `Tr(T_0)=Tr(T_1)=Tr(T_2)`,
`Tr(D_0)=Tr(D_1)=Tr(D_2)`, and `Tr T_0 ≠ Tr D_0`. The last must be
checked outside `a1d3_kalg`'s reducer, whose length-1 base case returns
the same symbol for every index by construction; it is verified on an
independent `SU2BPSKAlgebra` chart.)*

## 3. The recursions

Six identities close the system. Each is a chain of cyclicity moves and
relation rewrites, in the style of `pentagon_kq_proof.md` §7.2.

### (A) lowering `b` in type I — `b ≥ 2`

```
U_{a,b} = 𝖖^{-2a} U_{a,b-2} + 𝖖^{-2a-1} U_{a+1,b-2}
```

*Derivation.* Cyclicity moves the final `D_0` to the front as
`ρ²(D_0) = D_2`; then `D_2 T_0 = 𝖖^{-2} T_0 D_2` carries it across the
`a` copies of `T_0`; then `D_2 D_0 = 1 + 𝖖⁻¹ T_0` (the third relation at
`i = 2`):

```
Tr(T_0^a D_0^b) = Tr(D_2 · T_0^a D_0^{b-1}) = 𝖖^{-2a} Tr(T_0^a · D_2 D_0^{b-1})
                = 𝖖^{-2a} Tr(T_0^a D_0^{b-2}) + 𝖖^{-2a-1} Tr(T_0^{a+1} D_0^{b-2}).  ∎
```

### (B) the type-I edge `b = 1`

```
U_{a,1} = 𝖖^{-2a} V_{a,1}
```

*Derivation.* `Tr(T_0^a D_0) = Tr(D_2 T_0^a) = 𝖖^{-2a} Tr(T_0^a D_2)`,
the last step being the same `𝖖`-commutation. Here a cyclicity step
*converts a type-I trace into a type-II one*; there is no analogue of
this in the pentagon. ∎

### (C) lowering type II — `a ≥ 1`, `b ≥ 2`

```
V_{a,b} = χ₁ V_{a-1,b-1} + 𝖖^{2a-1} V_{a-1,b-2} + 𝖖^{2a} V_{a,b-2} + 𝖖^{1-2a} V_{a-1,b}
```

*Derivation.* `ρ²(D_2) = D_1`, and `D_1` does **not** `𝖖`-commute with
`T_0`: `D_1 T_0 = χ₁ + 𝖖 D_0 + 𝖖⁻¹ D_2`. Carrying the surviving `D`'s
across `T_0^{a-1}` with `D_0 T_0 = 𝖖² T_0 D_0`,
`D_2 T_0 = 𝖖^{-2} T_0 D_2`, and finishing with `D_0 D_2 = 1 + 𝖖 T_0`:

```
V_{a,b} = Tr(D_1 T_0^a D_2^{b-1})
        = χ₁ V_{a-1,b-1} + 𝖖^{2a-1} Tr(T_0^{a-1} D_0 D_2^{b-1}) + 𝖖^{1-2a} V_{a-1,b}
        = χ₁ V_{a-1,b-1} + 𝖖^{2a-1}(V_{a-1,b-2} + 𝖖 V_{a,b-2}) + 𝖖^{1-2a} V_{a-1,b}.  ∎
```

### (D) the type-II edge `b = 1` — `a ≥ 1`

```
V_{a,1} = χ₁ U_{a-1,0} + (𝖖 + 𝖖^{1-2a}) V_{a-1,1}
```

*Derivation.* The same first step as (C) with `b = 1`:
`V_{a,1} = χ₁ U_{a-1,0} + 𝖖^{2a-1} U_{a-1,1} + 𝖖^{1-2a} V_{a-1,1}`, then
(B) turns `U_{a-1,1}` into `𝖖^{-2(a-1)} V_{a-1,1}`, and
`𝖖^{2a-1}·𝖖^{2-2a} = 𝖖`. ∎

This is the cleanest of the six: a one-index recursion in `a`, seeded by
`V_{0,1} = Tr D_0`.

### (E) the pure-`T` tower — `a ≥ 2`

```
U_{a,0} = U_{a-2,0} + 𝖖^{3-2a} χ₁ V_{a-2,1} + 𝖖^{6-4a} V_{a-2,2}
```

*Derivation.* `ρ²(T_0) = T_2`, so `Tr(T_0^a) = Tr(T_2 T_0^{a-1})`, and
`T_2 T_0 = 1 + 𝖖⁻¹ χ₁ D_2 + 𝖖⁻² D_2²` (the first relation at `i = 2`).
Carrying `D_2` and `D_2²` across `T_0^{a-2}` costs `𝖖^{-2(a-2)}` and
`𝖖^{-4(a-2)}`. ∎

### (F) the `a = 0` base

```
V_{0,b} = U_{0,b}
```

immediately from `Z_3`-invariance (§2.2), since with no `T` letter the
two cone types differ by `ρ`.

*(Certificate C3: (A) 36/36, (B) 8/8, (C) 20/20, (D) 8/8, (E) 8/8,
(F) 9/9, all as exact identities of reduced expressions.)*

### 3.1 The system closes

Give `U_{a,b}` and `V_{a,b}` the weight `4a + 3b`, breaking ties with
`U` before `V`. Then every right-hand side above is strictly smaller:
(A) drops the weight by 6 and 2; (C) by 7, 10, 6, 4; (D) by 4 and 1;
(E) by 8, 5, 2; and (B) alone ties the weight, dropping instead in the
type tiebreak `U → V`, which cannot recur because (C) and (D) never
return to type I at equal weight. So the system terminates, with base
`U_{0,0} = Tr 1`, `U_{1,0} = Tr T_0`, `U_{0,1} = Tr D_0`.

**No pentagon-style collapse exists.** In the pentagon,
`ρ²(L_{i+1}) = L_i` makes `Tr L_i^a L_{i+1}^b = Tr L_i^{a+1} L_{i+1}^{b-1}`
a pure relabelling, folding the two indices into `a+b` and leaving a
one-index recursion with two seeds. Here `ρ²(D_i) = D_{i-1} ≠ T_i`, so
no such fold is available; the recursion is irreducibly two-index and
shuttles between the two cone types via (B), (D) and (E). **That is the
structural reason there are three elementary traces rather than two.**

## 4. Termination in general

§3 covers cone monomials. For an arbitrary word in the six letters — as
produced *inside* a reduction — the same conclusion holds, by two
lemmas about the letter alphabet alone.

**Lemma 4.1 (a decreasing weight).** For a word `w`, let `n_T`, `n_D`
count its `T`- and `D`-letters. Every relation rewrite replaces two
adjacent letters by the outputs above, and

```
S(w) = 4 n_T(w) + 3 n_D(w)
```

strictly decreases on **every** output term, by at least 2. `S` is
additive over concatenation, so the pairwise statement lifts to whole
words; and `S` is unchanged by `𝖖`-commutations (transpositions) and by
`ρ`-relabelling (which preserves letter kind), the two moves that run
between rewrites. Hence the recursion depth is at most `⌊S/2⌋`.

*The admissible weights are exactly `w_D < w_T < 2 w_D`*: `T_iT_{i+1} → D_i²`
forces `w_T > w_D`, and `D_iD_{i+1} → T_{i+1}` forces `2w_D > w_T`. The
lexicographic pair `(2n_T + n_D,\ n_D)` also works but has order type
`ω²` and yields only a quadratic depth bound. *(Certificate C5: both
drop on all 48 output terms.)*

**Lemma 4.2 (a rewrite always fires within 3 cyclicity steps).** The
letters `𝖖`-commuting with a given one are

```
comm(T_i) = {T_i, D_i, D_{i-1}}        comm(D_i) = {D_i, T_i, T_{i+1}}
```

(the relation is symmetric: `𝖖`-commutation fails exactly when the
unordered pair is `TT`, `DD`, or `T_i D_{i+1}`). Three cyclicity steps
run the moved letter through its whole `ρ`-orbit while the multiset of
the remaining letters is unchanged, so a stall would need every other
letter to lie in

```
⋂_i comm(T_i) = ∅        or        ⋂_i comm(D_i) = ∅ ,
```

both empty. Hence a rewrite fires. **The bound `3 = ord(ρ)` is sharp**:
the pairwise intersections are singletons — `comm(T_0) ∩ comm(T_1) = {D_0}`,
`comm(D_0) ∩ comm(D_2) = {T_0}`, and their `Z_3`-images — so words such
as `D_1^k T_0` do need all three steps.

*Whether a word stalls depends only on its last letter and the **set** of
its other letters*, so `6 × (2⁶ − 1) = 378` cases exhaust **every word of
every length**. *(Certificate C5: 378 cases, 0 survivors, exactly 6
needing all three steps.)*

**Theorem 4.3 (reduction).** Every `Tr(L_c)` is a
`Z[𝖖,𝖖⁻¹]⊗R_{SU(2)}` combination of `Tr 1`, `Tr T_0`, `Tr D_0`.

*Proof.* Lemma 4.2 supplies a rewrite, Lemma 4.1 makes the recursion
well-founded, and the base cases are the empty word (`Tr 1`) and the
six length-one words, collapsed to two by §2.2. Soundness is immediate:
`𝖖`-commutations and rewrites are relations *in the algebra*, so they
preserve the element; the cycling step is the cyclicity axiom. ∎

*(Certificate C6: exhaustively over all 9331 words of length ≤ 5 the
reduction returns only the three seed symbols. Independently, all 55 987
words of length ≤ 6 and samples to length 24 behave the same.)*

## 5. `ρ²`-cyclicity forces no relation among the seeds

Theorem 4.3 bounds the space of such functionals by 3. That the bound is
attained is the `[A_1,D_3]` analogue of the pentagon's five families.

**Lemma 5.1 (cyclicity need only be tested on generators).** Suppose
`Tr(x g) = Tr(ρ²(g) x)` for every generator `g` and every canonical `x`.
Then `Tr(uv) = Tr(ρ²(v)u)` for all `u, v`.

*Proof.* Induction on the length of `v` as a word in generators.
For `v = v'g`,

```
Tr(uv) = Tr((uv')g) = Tr(ρ²(g)(uv')) = Tr((ρ²(g)u)v')
       = Tr(ρ²(v')ρ²(g)u) = Tr(ρ²(v'g)u) = Tr(ρ²(v)u),
```

the first step by hypothesis, the fourth by the induction hypothesis on
`(ρ²(g)u,\ v')`, using associativity and that `ρ²` is an algebra map. ∎

**Theorem 5.2.** Cyclicity imposes no linear relation on
`Tr 1`, `Tr T_0`, `Tr D_0`; the space of `ρ²`-twisted-cyclic
`Z[𝖖,𝖖⁻¹]`-linear functionals is exactly three-dimensional.

*Proof.* By Lemma 5.1 it suffices to check `Tr(ρ²(g)·X) = Tr(X·g)` for
`g` a generator and `X` a canonical monomial. `Z_3`-invariance fixes the
generator, leaving `g ∈ {T_0, D_0}` and `X` ranging over the six cones:
**twelve families**, against the pentagon's five. Each closes
*identically* — both sides reduce to the same combination of the three
seeds, with nothing left over — so no constraint is produced.
*(Certificate C4: 12 families × 49 values of `(a,b)` = 588 checks, 0
mismatches.)* Consequently the three seeds are free, and with Theorem 4.3
the dimension is exactly 3. ∎

**Corollary 5.3.** Any linear relation among the classes would push
forward through `Tr` to a relation among the three values. None exists:
an exact search over `Z[𝖖^±]⊗R_{SU(2)}` finds nullity 0 at every window
tried (up to 975 unknowns, `𝖖`-degrees in `[-12,12]`, `χ` to `χ₁₂`,
equations through `𝖖³⁴`), with a planted-relation control correctly
detected.

## 6. Determination from the `O(𝖖)` axioms

As in the pentagon, the recursions carry **very negative powers of `𝖖`**,
and the axioms forbid them in the answer. The flavoured form of the
orthonormality axiom is that *every* character coefficient of
`I_{a,b} = Tr(ρ(L_a) L_b)` lies in `Z[[𝖖]]`, with `δ_{a,b} + O(𝖖)` on the
trivial character (`trace_bootstrap_status.md`, "Flavoured").

**The sharp leading constraint.** Since `ρ(T_{i-1}) = T_i`, the pairing
`Tr(ρ(T_{i-1})·T_i)` is `Tr(T_i²)`, and (E) with (F) and (A) gives the
exact reduction

```
Tr(T_i²) = (1 + 𝖖⁻²) Tr 1 + 𝖖⁻³ Tr T_0 + 𝖖⁻¹ χ₁ Tr D_0 .
```

`T_{i-1} ≠ T_i` as labels, so this must be `O(𝖖)` — the three poles must
cancel. Reading off orders `𝖖⁻²`, `𝖖⁻¹`, `𝖖⁰` with `Tr 1 = Σ c_n 𝖖ⁿ`,
`c_0 = 1`, `c_1 = 0`:

```
𝖖⁻² :  [𝖖¹] Tr T_0 = −1        so   Tr T_0 = −𝖖 + O(𝖖²)
𝖖⁻¹ :  [𝖖²] Tr T_0 = 0
𝖖⁰  :  [𝖖³] Tr T_0 + χ₁·[𝖖¹] Tr D_0 + 1 + c_2 = 0
```

the last consistent with `[𝖖¹] Tr D_0 = −χ₁` and `c_2 = χ₂` because
`χ₁² = 1 + χ₂`. This is the exact counterpart of the pentagon's
`Tr L_1 = (−𝖖 + O(𝖖⁴)) Tr 1`. *(Certificate C7.)*

**The two constraint families.** Iterating (E) and (A) expresses the two
towers as

```
Tr(T_0^a)      = A_a Tr 1 + B_a Tr T_0 + C_a Tr D_0
Tr(T_0^a D_0)  = A'_a Tr 1 + B'_a Tr T_0 + C'_a Tr D_0
```

with coefficients in `Z[𝖖^±]⊗R_{SU(2)}` whose `𝖖`-valuation falls off
quadratically in `a`. The first few rows:

| `a` | `A_a` | `B_a` | `C_a` |
|---|---|---|---|
| 2 | `𝖖⁻² + 1` | `𝖖⁻³` | `χ₁𝖖⁻¹` |
| 3 | `𝖖⁻⁷ + 𝖖⁻⁵ + (1+χ₂)𝖖⁻³` | `𝖖⁻⁸ + 𝖖⁻⁴ + 1` | `χ₁(𝖖⁻⁶ + 𝖖⁻⁴ + 𝖖⁻²)` |

and for the second family

| `a` | `A'_a` | `B'_a` | `C'_a` |
|---|---|---|---|
| 1 | `χ₁𝖖⁻²` | `0` | `𝖖⁻³ + 𝖖⁻¹` |
| 2 | `χ₁(𝖖⁻⁷ + 𝖖⁻³)` | `χ₁𝖖⁻⁴` | `𝖖⁻⁸ + 𝖖⁻⁶ + 𝖖⁻⁴ + 𝖖⁻²` |

Each row is a canonical basis element, so each lies in `R_{SU(2)}[[𝖖]]`;
every negative order therefore contributes a linear equation on the
three unknown series. The two families are linearly independent — the
first has only even `χ` on `Tr 1`, `Tr T_0` and odd on `Tr D_0`, the
second the reverse — and as `a` grows the constraints reach ever deeper,
determining `Tr T_0/Tr 1` and `Tr D_0/Tr 1` to all orders, leaving only
the overall `1 + O(𝖖)` rescale the contract permits.

**This is verified constructively.** `experiments/smart_trace_bootstrap.py::
flavoured_cone_solve` runs exactly this bootstrap — cyclicity plus
per-character `Z[[𝖖]]` positivity, *with no support typing and no frozen
tables* — and reproduces all three seeds from `Tr 1` alone, matching the
closed forms of the definition:

```
Tr 1    = 1 + χ₂𝖖² + (χ₀+χ₂+χ₄)𝖖⁴ + (χ₀+2χ₂+χ₄+χ₆)𝖖⁶ + …
Tr T_0  = −χ₀𝖖 − χ₂𝖖⁵ − (χ₀+χ₂)𝖖⁷ − …
Tr D_0  = −χ₁𝖖 − χ₃𝖖³ − (χ₁+χ₃+χ₅)𝖖⁵ − …
```

So the chiral-algebra closed forms in terms of `sl(2)_{−4/3}` characters
are a *check* on the axiomatic determination, not an input to it —
exactly the pentagon's relation to `M(2,5)`.

## 7. Certificate

```bash
PYTHONPATH=. python scripts/verify_a1d3_trace_recursions.py
```

C1 relation table (36 ordered pairs) · C2 `Z_3`-invariance and
`R`-linearity · C3 the six recursions (89 instances) · C4 the twelve
generator families (588 checks) · C5 termination (48 output terms; the
378-case no-stall) · C6 reduction to the three seeds (9331 words) ·
C7 the leading `O(𝖖)` constraint, both constraint tables of §6, and
their opposite `χ`-parity.

## 8. Two defects found while writing this — both fixed

**A silent-drop defect in `trace` (fixed).** `_trace_reduce_word` carried
a hardcoded `max_depth=40` while, by Lemma 4.1, the recursion depth grows
linearly in the word length (measured: exactly `2(n-1)` for a word of
length `n`). Past the cap the reducer emits a `('Tr_max_depth', …)` key
— and `A1D3KAlg.trace` iterates over the three seed keys only, so those
terms were **silently discarded** and a wrong series returned with no
exception. Demonstrated by reducing `T_0^6` under a deliberately low
cap: 11 failure-keyed terms vanish and the `Tr 1` coefficients diverge
from the correct ones at `𝖖⁻¹⁴`; the cap bit for genuine labels around
`a + b ≈ 22`. `trace_layer1` now sizes its guard from Lemma 4.1's bound
`⌊(4 n_T + 3 n_D)/2⌋` and **raises** on any non-elementary key instead
of dropping it. Regression test:
`tests/test_a1d3_trace_recursions.py::test_trace_layer1_raises_on_stray_key`.

**A gap in the sibling family's termination proof (fixed).**
`a1a2k_kq_proof.md` Proposition 6.1(ii) rested on Lemma 2.1's crossing
count `cr(w)`, which Lemma 2.1 shows decreases under Ptolemy splices —
but `cr` is **not rotation-invariant**: re-tagging a single chord by `ρ²`
can *raise* it by 2 (heptagon), and the trace engine's frames are
separated by exactly such re-tags. The repair is the `[A_1,D_3]`
insight applied there: use a measure the rotation cannot see. Chord
*length* is rotation-invariant, and the concave weight `w = d(H−d)` gives,
for a quadrilateral with arc gaps `p,q,r,s ≥ 1`,

```
    W_in − W_1 = 2qs ,        W_in − W_2 = 2pr
```

— both `≥ 2`, so every splice strictly decreases `W` while the re-tags
and q-commutations leave it fixed. That is now `a1a2k_kq_proof.md`
Lemma 2.1b, with Prop 6.1(ii) citing it; `cr` remains correct for §2's
multiplication reducer, which never rotates a letter. Concavity is
load-bearing — the linear weight `w(s) = s` ties on the second output —
and both facts are certified, with controls, in
`scripts/verify_a1a2k_kq_proof.py::run_length_weight`.
