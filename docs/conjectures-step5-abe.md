# Conjectures — Step 5 (the abelianized presentations)

The abelianized layer (`src/abe/`) presents the K-theoretic Coulomb branch algebras
of conventional gauge theories on an enriched rational quantum torus, and ties every
presentation together through the object layer. Three structural statements are the
ones it directly bears on — and, in the constructive spirit of the project, it does
not merely *check* them, it *uses* them to build and to organise the algebras.

## 1. Orthonormality of the canonical basis

**Conjecture.** The abelianized algebras presented in `src/abe/` satisfy the
`K_𝖖`-algebra axioms; in particular, for the canonical basis `{L_a}` the Schur
pairing

    I_{a,b}(𝖖)  =  Tr( ρ(L_a) · L_b )   satisfies   I_{a,b}(𝖖) = δ_{a,b} + O(𝖖) :

the canonical basis is **orthonormal to leading order in `𝖖`**. (The `𝖖⁰` term is
the `Δ = spin = 0` identity sector — see `docs/conjectures-step1-samples.md`.) On
the enriched torus the pairing is the Schur-measure residue; for a matter theory
the statement is on the identity summand of the flavour character.

## 2. The canonical basis is genuine — the constructive-build discipline

A canonical element is assembled **constructively**, as a polynomial in
already-built bar-invariant generators (dressed minuscules, `det`, Wilson lines),
never solved for. A solve is barred on this tier because it can land *outside* the
canonical span, and both orthonormality and the internal span-pairing test are
pairings defined *within* the span, so both would pass on off-span content.
Consequences that are enforced: every canonical is bar-invariant by construction;
every decomposition is certified by exact reconstruction; acceptance is
`certify_canonical(a) == a` (bar-palindromicity together with the single-leading
q-extreme condition); where a closed form does not reach, the tier raises rather
than guesses. (In type A every canonical is a polynomial of dressed minuscules, so
product-and-peel is complete.)

## 2b. Which `KAlgebra` axioms are *derived*, and the one that is not

Most of §1 and §2 above is not, in fact, conjectural on this tier: once the
substrate is fixed, all but one of the `KAlgebra` axioms follow **analytically**
from the tier's own axioms — the W2 seed, bar (`W1`), `O(𝖖)` and (★). A verifier
that passes them is confirming arithmetic rather than probing the mathematics,
and knowing which is which is the point of this section.

| `KAlgebra` axiom | status | where the content is |
|---|---|---|
| unit `L_{0,0}·L_a = L_a` | **derived** | `chart(L_{0,0}) = χ_0·U_0 = 1`, the torus unit |
| bar antimultiplicativity | **derived** | `W1` + `bar(CC_{a,b}) = CC_{b,a}` |
| `ρ` automorphism, `ρ⁻¹ρ = id` | **derived** | `ρ` is conjugation by the √measure — an automorphism of the substrate |
| `ρ²`-twisted trace cyclicity | **substrate** | a property of the measure residue; no canonical-basis input at all |
| orthonormality `I_{a,b} = δ + O(𝖖)` | **derived** | `W2` + `O(𝖖)` + two substrate lemmas |
| associativity | **derived, given closure** | the torus product is associative and `chart` is faithful |
| **closure** — `L_a·L_b ∈ span_{Z[𝖖^{±1}]}{L_c}` | **NOT derived** | the open question, below |

**Bar, in one line.** `bar` is an antiautomorphism of the substrate *because* the
cocycle satisfies `bar(CC_{a,b}) = CC_{b,a}` (this is a property of `CC`, not of
the derived `R`). So `bar(x·y) = bar(y)·bar(x)`; taking `x = chart(a)`,
`y = chart(b)`, both bar-invariant by `W1`, gives
`bar(chart(a)·chart(b)) = chart(b)·chart(a)`, and reading that back through the
linearly independent `L_c` yields `C^c_{ab}(𝖖⁻¹) = C^c_{ba}(𝖖)`. No numeric
input.

**Orthonormality, from `W2` + `O(𝖖)`.** `I_{a,b}` is the magnetic-0 residue of
`ρ(x)·y`, and since `ρ` reverses the magnetic charge the sum is
**sector-diagonal**, pairing `x_n` against `y_n` through a weight `w_n`. Two
lemmas about the *substrate* finish it: the weight costs nothing
(`val_𝖖 w_n = 0` — the Vandermonde's negative power is cancelled exactly by the
cocycle), and the leading orbits are orthogonal (the orbit sum reduces to
orthonormality of the Levi characters). Then `W2` + `O(𝖖)` say `val_𝖖 x_n = 0`
exactly on the leading orbit and `≥ 1` elsewhere, so `I_{a,b}[𝖖⁰]` can only
receive contributions where the two orbits meet — nothing when the dominant
magnetic labels differ, and the pairing of the leading orbits alone when they
agree.

**A corollary that explains a known measurement.** The derivation shows
`I[𝖖⁰]` **cannot see the bubbling at all**: every non-leading cell is `O(𝖖)` by
axiom and the weight cannot lower it. So orthonormality at `𝖖⁰` is *blind* to a
candidate missing its bubbling — which is exactly why a bar-blind `𝖖⁰` self-norm
once failed to catch an off-span solve while (★) rejects that content 23/23. The
two facts are the same fact, and it is the sharpest statement of why (★) is a
guard of a different kind: a membership condition on `Λ = R(G)`, not a pairing
inside the span.

⚠ A label handed over as a **non-dominant representative** is not a distinct
label. At SU(3), `dominant_cochar_rep((1,0)) = (1,1)`, so `L_{(1,0),0}` and
`L_{(1,1),0}` are the same line and their pairing is `1`, not `0`. A δ-test on
raw tuples reports spurious off-diagonal ones.

**Closure — the one that is not derived.** One containment is immediate: (★) says
each `L_{m,e}` is a `𝖖`-difference operator preserving `Λ = R(G)`, and such
operators are closed under composition, so `L_a·L_b` always lies in the algebra
of (★)-satisfying operators. The algebra axiom is **exactly** the converse — that
the canonicals *span* that algebra:

    {operators satisfying (★)}  ⊆  span_{Z[𝖖^{±1}]}{L_c}.

That is open. Until it is proved, closure is **measured**, and the measurement
has to be on products and by **exact reconstruction** —
`chart(a)·chart(b) == Σ_c C^c_{ab}·chart(c)` on the torus — not merely "`multiply`
returned something", because the decomposition peels top-down and would return a
*partial* answer silently if it ever stopped early.

## 3. One abstract algebra, many certified presentations

An abstract `A_𝖖[T]` admits several presentations — a BPS-quiver chart, a cone, an
RG flow, the abelianized quantum torus — and a `KAlgebraIso` certifies that two of
them are the same object (a bijective canonical-label map intertwining `multiply`,
`ρ`, and the trace). The object layer (`KAlgebraObject`) holds many presentations
under one roof with a **path-independence (coherence)** certificate: the transition
isomorphisms compose consistently around every loop.

### The RG-intertwining relation (how matter is built)

The abelianized presentation of a **matter** theory is a pure-gauge `AbeKAlgebra`
combined with an `RGKAlgebra` flow that adjoins the matter (`UNNfOverPure`). The
flow relation

    RG(a)·S_RG  =  S_RG·ρ_IR⁻¹(RG(ρ_UV(a)))  =  L_{apex(a)} + O(𝖖)

(see `docs/conjectures-step3-rg.md`) is *used* to build the flow, and the resulting
algebra is certified `≅` its BPS-quiver realisation. The N=2\* machinery
(`N2StarBuilder`) is the same idea applied physically: solve in N=2\*, then RG-flow
to read off the pure-U(N)/SU(N) canonical `L_{m,e}`.

## Verification scope

What the tests actually certify (`tests/test_abe_flows.py`, ~7.5 s):

| check | scope |
|---|---|
| pure-U(N) keystone contract + orthonormality | `PureUNKAlgebra(2)`; the KAlgebra axiom battery, `I_{a,b}=δ+O(𝖖)` |
| matter as native Abe + the flow ↔ BPS iso | `UNNfKAlgebra(2,1)`; full `KAlgebraIso` battery (unit / round-trip / multiplicative / ρ / trace) at U(2)+N_f=1 |
| N=2\* canonical finder + SU(2) N=2\* flow | `N2StarBuilder(2)` decomposes a torus slice to a single canonical; `SU2N2StarRGKAlgebra` exact-FS trace |
| object layer | `pure_u2_object` / `u2_nf1_object`: `abe↔cone` full battery, `abe↔bps` multiply sector, `verify_coherence`; `pure_un_bps_iso(2)` |
| SU(2) family | native `PureSU2KAlgebra` orthonormality; `pure_su2_object` / `su2_nf1_object` (with RG-flow legs to pure SU(2) and to the pentagon) / `su2_unf_object` (Spin(2N_f) enhancement), each with pairwise batteries + coherence |
| SU(3) / U(3) | native `PureSUNKAlgebra(3)` orthonormality; `pure_un_bps_iso(3)`, `pure_sun_bps_iso(3)` |
| flow-typed object layer | `subquiver_flow_object` (F-oracle vs co-solver, bridged by an `RGKAlgebraIso`); `su2a1d3_rg_object` / `su2a1d4_rg_object`, incl. the Abe + RG object |
