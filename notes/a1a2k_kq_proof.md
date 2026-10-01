# The `[A_1, A_{2k}]` chord algebras are K_𝖖-algebras — K1–K3 proofs and the trace program

**Objects.**  `A1A2kKAlg(k)` (`implementations/a1a2k_kalg.py`) — the
stand-alone realisations of `A_𝖖([A_1, A_{2k}])`, `k ≥ 1`: the
(2k+3)-gon **chord algebras**.  `k = 1` is the pentagon
(`pentagon_kq_proof.md` proves all axioms for its `PentagonKAlg`
presentation); this document treats `k ≥ 2` — concretely the
**heptagon** (`k = 2`) and **nonagon** (`k = 3`) — by the same method.
As for the pentagon, no BPS machinery enters the *definition*: the
class is a `ConeKAlgebra` over a closed-form chord-geometry table
(`implementations/A1A2k_plucker_closed_form.py`), with Andrews–Gordon
Layer-2 trace seeds (`implementations/minimal_model_characters.py`).
The BPS chart is used below only as the *source of a witness* for the
freeness certificate; every fact about the witness is re-verified
independently at certificate time.

**Status.**
* **(K1) free canonical basis, associativity, multiply bridge** —
  **proven** for `k = 2, 3` (general-k lemmas + per-k finite
  certificates, §§3–4).
* **(K2) bar involution** — **proven** for `k = 2, 3` (general lemma +
  per-k finite mirror certificate, §5).
* **(K3) ρ** — **proven** for all `k` (structural, §5).
* **(K4a) trace cyclicity** — the two-layer trace engine is proven
  valid *for any* ρ²-twisted-cyclic functional with the implemented
  seeds, and terminating (§6); the **existence** of such a functional
  (the heptagon analogue of the pentagon's explicit `T`) is the open
  step, with the supporting structure mapped: the **tower closed
  forms** in 2-fold Andrews–Gordon tails (§7, certified to deep
  windows), the uniform AG contiguous relation (proven, §7.1), and the
  new finding that the trace is AG-tail-valued on whole cone families
  but needs an enlarged family on mixed-long-chord monomials (§7.4).
* **(K4b) orthonormality** — machine-certified windows; program as for
  the pentagon (§8).

Certificate: `scripts/verify_a1a2k_kq_proof.py` (exact arithmetic).
Conventions: `q = 𝖖`, `p = q²`, Gaussian binomials in base `q²`.

---

## 1. The implemented data (general `k`)

Fix `k ≥ 1`, `H := 2k+3`.

* **Generators.**  `L_{(a,i)}`, `a ∈ {1,…,k}`, `i ∈ Z/H` — the chord
  with endpoints `(i, i+a+1)` on the `H`-gon (all diagonals; edges
  `(i,i+1)` are *not* generators — they are normalised to `1`).
* **Canonical labels 𝔅_k.**  Sorted tuples
  `((a_1,i_1,e_1),…,(a_m,i_m,e_m))`, `e_r ≥ 1`, with **pairwise
  non-crossing** chords; `()` is the identity.  Non-crossing chord
  sets = partial triangulations; the maximal ones (triangulations,
  `2k` chords each) are the cones.
* **Relations (the closed-form table).**  For two chords:
  - *same / share / disjoint* (non-crossing): q-commutation
    `L_g L_h = q^{2c(g,h)} L_h L_g` with the integer cocycle from the
    arc-parity rule (`_qc_arc_parity`);
  - *crossing*: the **quantum Ptolemy relation** — for the
    quadrilateral `a<b<c<d` with diagonals `(ac),(bd)`:
    `L_{ac} L_{bd} = q^{α}·[L_{ab}L_{cd}] + q^{β}·[L_{ad}L_{bc}]`,
    edges replaced by `1`, with `(α,β)` given by the arc-parity rule of
    `A1A2k_plucker_closed_form.py` and the bracketed products written
    in canonical (X-basis) form.
* **Normalisation.**  `X_label = q^{−T(label)}·(sorted L-product)`,
  `T(label) = Σ_{r<s} c_fwd(g_r,g_s) e_r e_s` — the universal
  `cone_label_phase` convention, exactly as the pentagon's `q^{ab}`.
* **ρ** — the polygon rotation `i ↦ i+1` on all letters.
* **multiply** — the generic `ConeKAlgebra` reducer over
  `A1A2kConeData`.
* **trace** — Layer 1 (generic tagged-cyclicity engine) to the seeds
  `{(), ((a,0,1),) : a = 1..k}`, then Layer 2:
  `T_0 = χ_1(q²)`, `T_a = (−1)^{m+1}q^{−m}(χ_m − χ_{m+1})(q²)` with
  `m = k−a+1` and `χ_s = χ_s^{(2,2k+3)}` the Andrews–Gordon characters.

**Definition 1.1.**  `P_k` := the unital associative
`Z[q,q⁻¹]`-algebra presented by the `kH` chord generators and the
relations above (one q-commutation per non-crossing pair, one Ptolemy
per crossing pair — both orders; a finite presentation).
`𝔅_k ⊂ P_k` is the candidate basis
`X_label := q^{−T(label)}·∏ L^{e}` (sorted order).

---

## 2. General lemma — spanning and termination

**Lemma 2.1 (crossing measure).**  Let `w` be a word in the chord
generators and `cr(w)` its total number of crossing letter pairs.
Every Ptolemy splice strictly decreases `cr`; q-commutation swaps and
merges preserve it and the length.

*Proof.*  Resolving the crossing pair `{(ac),(bd)}` of a quadrilateral
`a<b<c<d` into `{(ab),(cd)}` or `{(ad),(bc)}` removes that crossing.
For any other chord `e`: `e` crosses a chord `(uv)` iff `e` separates
`u` from `v` on the circle.  Checking the five possible splittings of
`{a,b,c,d}` by `e` (`∅`, one vertex, `ab|cd`, `ad|bc`; the interleaved
split `ac|bd` is impossible for a chord): the number of crossings of
`e` with each replacement pair never exceeds its crossings with
`{(ac),(bd)}`.  Hence `cr` drops by at least 1.  ∎

> **`cr` is the right measure for the MULTIPLICATION reducer only.**  It
> is *not* rotation-invariant: rotating a single chord can raise `cr` by
> 2 (measured in the heptagon).  `derived_multiply` never rotates a
> letter, so Proposition 2.2 below is unaffected; but the **trace**
> engine does — its cyclicity step re-tags one letter by `ρ²` between
> splices — so §6 needs the rotation-invariant measure of Lemma 2.1b
> instead.  Using `cr` there was an error, corrected 2026-08-30.

**Lemma 2.1b (length weight — rotation-invariant).**  For a chord with
endpoints `x, y` on the `H`-gon put

```
    w = d(H − d),        d = (y − x) mod H
```

(well defined: `d(H−d) = (H−d)(H−(H−d))`, so `w` depends only on the
chord's length), and let `W(w)` be the sum over the letters of a word.
Then `W` is **invariant** under rotating any single letter and under
q-commutation swaps, and **strictly decreases by at least 2** at every
Ptolemy splice.

*Proof.*  Invariance is immediate: a rotation `x ↦ x+1, y ↦ y+1` fixes
`d`, and a swap permutes the letters.  For the splice, let the
quadrilateral `a<b<c<d` have arc gaps `p, q, r, s ≥ 1`,
`p+q+r+s = H`.  Then

```
W_in = w(ac) + w(bd) = (p+q)(r+s) + (q+r)(s+p) = 2pr + ps + qr + 2qs + pq + rs
W_1  = w(ab) + w(cd) = p(q+r+s) + r(s+p+q)     =  pq + 2pr + ps + rs + qr
W_2  = w(ad) + w(bc) = (p+q+r)s + q(r+s+p)     =  ps + 2qs + rs + qr + pq
```

so

```
    W_in − W_1 = 2qs ,        W_in − W_2 = 2pr ,
```

both `≥ 2` because `p, q, r, s ≥ 1`.  A side that is an **edge** is
replaced by `1`, dropping its weight from the output entirely, which
only widens the gap.  ∎

*(The two identities are exactly why a **linear** weight fails: for
`w(s) = s` the second output ties.  Concavity is doing the work, and
`d(H−d)` is the natural rotation-invariant concave choice.  Checked on
all 5984 quadrilaterals with `H = 5..21`, and on every Ptolemy splice
for `H = 7, 9, 11, 13, 15` — uniform minimum drop 2.)*

**Proposition 2.2 (spanning).**  𝔅_k spans `P_k`.  Moreover the
generic reducer (`derived_multiply` over `A1A2kConeData`) terminates
on every input and outputs a `Z[q,q⁻¹]`-combination of 𝔅_k.

*Proof.*  Induction on `(cr(w), len(w))` lexicographically.  If
`cr = 0` the letters are pairwise non-crossing: sort by q-commutation
into a canonical word `∈ q^{Z}𝔅_k`.  Otherwise pick a crossing pair at
minimal index distance; by minimality the letters between them
q-commute with the left one (any letter crossing it would give a
closer pair), so finitely many swaps make the pair adjacent and the
Ptolemy splice applies; each daughter has smaller `cr` (Lemma 2.1).
Termination of the implemented reducer: same multiset argument as the
pentagon (Theorem 3.6 there) with the ordinal
`ω²·cr(w) + ω·len(w) + ε(w)`; the cone-data conventions
(`cocycle = table/2`, `cross_product` = the table with the
`cone_label_phase` correction, wrap conventions) are checked by the
certificate against the closed-form table.  ∎

---

## 3. K1 — freeness via a quantum-torus witness

Let `Q_k := Q_𝖖(Z^{2k})` with `X_γX_δ = q^{⟨γ,δ⟩}X_{γ+δ}`,
`⟨γ,δ⟩ := γ·B·δ` for the `A_{2k}` path pairing `B`
(`B_{a,a+1} = +1 = −B_{a+1,a}`).  Let `N := Z_{≥0}^{2k}` (the
node-charge cone) with the partial order `γ ⪯ δ ⟺ δ−γ ∈ N`.

**Definition 3.1 (witness).**  An **F-witness** is an assignment
`g ↦ F(g) ∈ Q_k` (finite support, `Z[q,q⁻¹]` coefficients) to the
`kH` generators such that:

* **(F1)** every defining relation of `P_k` holds among the `F(g)` in
  `Q_k` (all `(kH)²` ordered generator products);
* **(F2)** each `F(g)` has a unique ⪯-minimal support point `γ₋(g)`
  with coefficient exactly `1`, and `supp F(g) ⊆ γ₋(g) + N`;
* **(F3)** for every q-commuting (non-crossing) pair,
  `c_fwd(g,h) = ⟨γ₋(g), γ₋(h)⟩`, where `c_fwd` is the table's forward
  exponent (`L_gL_h = q^{c_fwd}·X[g+h]`-convention);
* **(F4)** the assignment `label ↦ γ₋(label) := Σ e_r γ₋(g_r)` is a
  bijection `𝔅_k → Z^{2k}`.  Certified form: the maximal cones
  `C = cone(γ₋(g) : g ∈ T)`, `T` a triangulation, are simplicial
  **unimodular**; every facet (codim-1 face) of a cone is shared by
  **exactly two** cones lying on opposite sides of its hyperplane; and
  one generic point is covered exactly once.  (Then the covering
  degree is locally constant on the complement of the codim-2
  skeleton, hence ≡ 1: the cones tile `Z^{2k}`, and unimodularity
  makes `(support, powers) ↦ lattice point` bijective, faces being
  consistent across neighbouring cones.)

**Theorem 3.2 (K1).**  If an F-witness exists then:
(i) `F` extends to an algebra map `P_k → Q_k`;
(ii) every `F(X_label)` has unit coefficient at `X_{γ₋(label)}` and
support in `γ₋(label) + N` *(by (F2) + (F3): the label phase
`q^{−T(label)}` exactly cancels the pairing phases of the bottom
product, as in the pentagon's Lemma 3.4)*;
(iii) 𝔅_k is `Z[q,q⁻¹]`-linearly independent *(minimal-bottom
argument verbatim from the pentagon's Theorem 3.5, using (F4))*;
hence with Proposition 2.2, **`P_k` is free with basis 𝔅_k, the
structure constants are well defined, and the implemented `multiply`
computes them** (uniqueness of coordinates in a free basis makes any
valid terminating reduction strategy compute the same output).  ∎

**Certificate (k = 2, 3).**  The script extracts the BPS-chart images
of the `kH` chords as the witness candidate and verifies (F1)–(F4)
exhaustively in exact arithmetic.  Results (`k = 2`): 14 generators,
supports of sizes 1–5, bottoms = the chord charges; all 28 base
products (= all 196 ordered pairs by ρ-lifting) hold in `Q_2` with the
`+1` sign convention; all 56 non-crossing pairs satisfy (F3); 42
cones (= the 42 triangulations of the 7-gon), all unimodular; 84
facets, each shared by exactly 2 cones, opposite-sided; generic point
covered once.  `k = 3`: same battery (27 generators, 429 cones).

**Corollary 3.3.**  `A1A2kKAlg(2)` (heptagon) and `A1A2kKAlg(3)`
(nonagon) satisfy K1.

---

## 4. K3 — ρ; the dihedral anti-automorphism

**Theorem 4.1 (K3).**  The rotation `L_{(a,i)} ↦ L_{(a,i+1)}` extends
to an algebra automorphism ρ of `P_k` of order `H`, fixing `1`,
permuting 𝔅_k exactly as the implemented `rho`/`rho_inverse`.

*Proof.*  The relation set is rotation-invariant by construction: the
chord-pair classification, the arc-parity cocycle and the Ptolemy
`(α,β)` depend only on the cyclic configuration, and the table is
*defined* by lifting base position 0 by ρ (rule (III) of the
closed-form module).  Labels: rotation preserves sortedness up to
re-sorting, non-crossing-ness, and the `T(label)` phase (`c_fwd` is
rotation-invariant).  ∎

**Theorem 4.2 (θ).**  The reflection `L_{(a,i)} ↦ L_{(a, −i−a−1)}`
(chord `(i,j) ↦ (−j,−i)`) extends to an anti-automorphism θ with
`θ² = id`, mapping 𝔅_k to 𝔅_k.  *(Reflections reverse cyclic order,
hence preserve the chord-pair classification while transposing
products; the certificate checks the finite table identity —
θ-image of the `(g,h)` entry = the `(θh,θg)` entry.)*

---

## 5. K2 — bar involution

**Lemma 5.1 (general).**  Suppose the finite table satisfies the
**mirror property**: for every ordered generator pair, the expansion
of `L_hL_g` equals that of `L_gL_h` with every exponent negated
(same canonical-basis terms).  Then `q ↦ q⁻¹` + word reversal
descends to an antimultiplicative involution `bar` of `P_k` fixing
every generator, and every element of 𝔅_k is bar-invariant:

```
bar(X_label) = q^{T(label)}·(reversed L-product)
             = q^{T(label)}·q^{−2T(label)}·(sorted L-product) = X_label ,
```

since reversing a sorted non-crossing word costs exactly
`q^{−2·Σ c_fwd e_re_s}` — wait, the reversal phases are
`Σ_{r<s} 2c(g_r,g_s)e_re_s = 2T(label) − 2Σ⟨…⟩`-free: precisely, with
`c(g,h) = c_fwd(g,h) − c_fwd(h,g) = 2c_fwd(g,h)`… the certificate
pins the integer identity `c(g,h) = 2·c_fwd(g,h)` (the table's
q-commute factors are even, asserted by `A1A2kConeData.cocycle`), so
the reversal cost is `q^{−2T(label)}` and the displayed cancellation
is exact.  Consequently `C^c_{uv}(q⁻¹) = C^c_{vu}(q)` for all pairs
(freeness + antimultiplicativity, as in the pentagon's Theorem 4.1).

**Certificate.**  The mirror property and the evenness
`c = 2c_fwd` are verified for all table entries at `k = 2, 3`.
**Hence K2 holds for the heptagon and nonagon.**  ∎

---

## 6. The trace engine (general `k`)

**Proposition 6.1.**  (i) Every Layer-1 move of
`simplify_trace_via_cone_data` over `A1A2kConeData` preserves the
value of the running expression under *any* `Z[q,q⁻¹]`-linear
functional `T` that is ρ²-twisted cyclic and linear — the moves are
relation rewrites (valid for any linear `T` by K1), cyclicity
instances, and the ρ²-orbit collapse.  (ii) The engine terminates: a
tagged chord, cycled through the `H` rotations, crosses any fixed
remaining chord at some rotation (two chords of lengths `≥ 2 ≤ k+1 <
H/2` always admit a crossing rotation), so a Ptolemy fires within
`cycle_period_bound = H` cycles, and **Lemma 2.1b's** measure decreases
— by at least 2, while the `ρ²` re-tags and the q-commutations between
splices leave it unchanged, so the recursion is well-founded with depth
at most `⌊W/2⌋`.  *(This cited Lemma 2.1's crossing count `cr` until
2026-08-30; `cr` is not rotation-invariant — a single re-tag can raise
it by 2 — so it cannot bound a recursion whose frames are separated by
rotations.  Lemma 2.1b is the repair; `cr` remains correct for §2's
multiplication reducer, which never rotates a letter.)*
(iii) Hence: **if** a cyclic functional with the implemented seed
values exists, the implemented `trace` computes it (with the exact
`K`-widening as in the pentagon's Theorem 7.5).  ∎

The open K4a step for `k ≥ 2` is the *existence* construction — the
analogue of the pentagon's explicit closed form on all of 𝔅.  §7
records how far that construction now reaches.

---

## 7. Andrews–Gordon tails and the trace towers (`k = 2`)

Define the **k-fold AG tails**: for shifts `(a_1,…,a_k) ∈ Z^k`,

```
R(a_1,…,a_k)(p) := Σ_{n_1,…,n_k ≥ 0}
    p^{N_1²+…+N_k² + a_1N_1 + … + a_kN_k} / ((p;p)_{n_1}⋯(p;p)_{n_k}) ,
    N_j := n_j + … + n_k .
```

The Andrews–Gordon characters are
`χ_s^{(2,2k+3)} = R(0,…,0,1,…,1)` (ones in slots `s..k`); `k = 1`
gives the pentagon's `R_s`.

### 7.1 The uniform contiguous relation

**Lemma 7.1.**  For all `k ≥ 1` and shifts `a`:

```
R(a_1,…,a_k)  =  R(a_1,…,a_{k−1}, a_k+1)
                 +  p^{k + a_1 + … + a_k} · R(a_1+2, …, a_k+2) .
```

*Proof.*  `R(a) − R(a+e_k) = Σ p^{…}(1−p^{N_k})/∏(p;p)_{n_i}`; since
`N_k = n_k`, replace `(1−p^{n_k})/(p;p)_{n_k} = 1/(p;p)_{n_k−1}` and
shift `n_k ← n_k−1`: **every** `N_j` drops by 1, so the exponent
gains `Σ_j (2N_j′+1) + Σ_j a_j = 2ΣN′ + k + Σa`, i.e. the summand of
`R(a+2·𝟙)` times `p^{k+Σa}`.  ∎

(`k = 1` is the pentagon's Lemma 6.1.  The certificate verifies
instances at `k = 1, 2, 3`.)

**Corollary 7.2 (seed identities).**  `χ_{k+1} − ... `: at `k = 2`,
`χ₂ − χ₃ = R(0,1) − R(0,0) = −p²R(2,2)` (Lemma 7.1 at `(0,0)`), so
the implemented orbit-1 seed is `Tr(L_{(1,·)}) = q²R(2,2)(q²)`;
the orbit-2 seed `Tr(L_{(2,·)}) = q⁻¹(χ₁−χ₂)(q²) = −q·R(1,2)(q²)`
holds by the **transposition identity**
`R(0,1) − R(1,1) = p·R(1,2)` (certified exactly; a slot-1 telescope —
its general-`k` form is part of the program).

### 7.2 The tower theorems (certified closed forms)

With `L₂ := L_{(2,0)}` (long chord) and `L₁ := L_{(1,0)}` (short
chord), the implemented heptagon trace satisfies, on every certified
window (through `q⁴⁴`, `a ≤ 6`; unique fits in the scanned family):

```
Tr(L₂^a)  =  (−1)^a q^{a}  · R(1, a+1)(q²)
Tr(L₁^a)  =        q^{2a} · R(a+1, a+1)(q²)
```

— the exact generalization of the pentagon's
`Tr(L^a) = (−1)^a q^a R_{a+1}(q²)` (`k = 1`: both laws degenerate to
it).  At `a = 0` both give `R(1,1) = χ₁^{(2,7)}` ✓; at `a = 1` they
reproduce §7.1's seed identities ✓.

*Status:* certified, not yet proven to all orders.  The proof
program: one cyclicity move on `L₂^a` (rotate one letter, Ptolemy
against `L_{(2,2)}`) couples the `L₂`-tower to the mixed towers
`Tr(L₂^m·L_{(1,j)})` — the `(F,G)`-system of the heptagon trace
miracle (`tests/test_heptagon_miracle.py`); all members fit AG tails
(§7.3), so the system should close under Lemma 7.1-type relations
exactly as the pentagon's single recursion did.

### 7.3 The trace is AG-tail-valued on whole cone families

Scanning all six ρ-orbit representatives of the 42 cones (exponents
≤ 2, ≥ 60 monomials): every monomial in cones avoiding a
*mixed-long-chord* pair fits a **unique** `± q^w R(a_1,a_2)(q²)`,
with shifts affine in the exponents (e.g. in the cone
`{(1,0),(1,2),(1,4),(2,4)}`: `a₁,a₂` grow by `(1,1)` per short-chord
unit and `(0,1)` per long-chord unit, matching §7.2).

### 7.4 …but not on mixed-long-chord monomials

For the cone `{(1,0),(1,2),(2,2),(2,6)}`, the monomials containing
**both** long chords (sharing a vertex) do *not* fit a single tail
`±q^wR(a_1,a_2)` — nor any single tail with an `N₁N₂` cross-term.
The two-tail law for the first stratum is certified (`q⁴⁰`,
`n := e_1+e_2 ≤ 4`, `min(e_1,e_2) = 1`):

```
Tr( X[L_{(2,2)}^{e_1} L_{(2,6)}^{e_2}] )
   =  (−1)^n q^n · [ R(2,n) + p^{n+1} R(3,n+1) ](q²) ,
```

while `min(e_1,e_2) = 2` (e.g. `e = (2,2)`) fits **no** two-tail sum —
the closed form of `T` on 𝔅₂ stratifies by `min(e_1,e_2)` along the
shared-vertex direction, with at least `min+1` tails per stratum and a
non-trivially re-based decomposition from `min ≥ 2` on (the residual
of the naive continuation starts `−p³+p⁴−p⁸+…`).  Pinning this family
(and the analogous structure at general `k`) is the missing K4a
existence step — recorded with data for the follow-up.

---

## 8. K4b status

`I_{u,v} = δ_{u,v} + qZ[[q]]` (with the strengthened negative
window): machine-certified on the §-certificate windows (`k = 2`:
all label pairs with exponents ≤ 1 across all cones, plus tower
pairs; `k = 3`: spot windows).  The pentagon-style proof (class
functions, per-class collapses, block-expansion recursion at the
diagonal) awaits K4a's explicit `T`.

## 9. Checklist

| axiom | heptagon (k=2) | nonagon (k=3) | general k |
|---|---|---|---|
| K1 free basis + associativity + bridge | **proven** (Thm 3.2 + certificate) | **proven** (same) | conditional on F-witness (Thm 3.2) |
| K2 bar | **proven** (Lemma 5.1 + certificate) | **proven** | conditional on mirror table |
| K3 ρ | **proven** (Thm 4.1) | **proven** | **proven** |
| K4a cyclic trace | engine valid+terminating (Prop 6.1); existence open — towers & seeds in AG form | same | program |
| K4b orthonormality | certified windows | spot windows | program |

The pentagon (`k = 1`) has all axioms fully proven in
`pentagon_kq_proof.md`; `A1A2kKAlg(1)` presents the same algebra in
the chord labelling (its towers match the pentagon closed form —
certified).
