# Conjectures — Step 4 (the BPS realisations)

The BPS layer (`src/bps/`) provides the BPS-quiver realisation engine: a concrete
`KAlgebra` built from a single IR chart (a BPS quiver + spectrum generator). The
two central statements of the framework are the ones it directly bears on — and,
in the constructive spirit of the project, the engine does not merely *check*
them, it *uses* them to **build** the algebra.

## 1. The `F_a · S = X_{γ_a} + O(𝖖)` discovery relation

For a BPS chart with spectrum generator `S` (the Kontsevich–Soibelman product
`S = ∏_i E_𝖖(X_{γ_i})`, with `E_𝖖(x) = (−𝖖x; 𝖖²)_∞^{−1}`), each canonical-basis
element is *discovered* as the unique `F_γ` whose image in the quantum torus
satisfies

    F_γ · S  =  X_γ + O(𝖖) ,    with bar-invariant coefficients.

This is the leading BPS special case of the general RG-intertwining relation
`RG(a)·S_RG = S_RG·ρ_IR⁻¹(RG(ρ_UV(a))) = L_{apex(a)} + O(𝖖)` (see
`docs/conjectures-step3-rg.md`).

**Constructive use here.** `BPSKAlgebra` *solves* the discovery relation for `F_γ`
(the F-finder, exact arithmetic in the localization `Z[𝖖^±][(1−𝖖^{2n})^{−1}, n≥1]`
over the doubly-tropical charge interval — see `docs/step4-BPSKAlgebra.md`),
then reads structure constants off the quantum torus: `F(L_a·L_b) = F(L_a)·F(L_b)`,
so `multiply` is multiply-in-the-easy-QT-then-recognise. The spectrum generator
may be supplied (the chamber spectrum) or **built from the quiver alone**
(the spec-free `build_S=True` constructor; §3). `ρ` on labels is the
closed-form piecewise-linear half-monodromy `σ`.

## 2. Orthonormality of the canonical basis

**Conjecture.** The BPS realisations presented in `src/bps/` satisfy the
`K_𝖖`-algebra axioms; in particular, for the canonical basis `{L_a}` the Schur
pairing

    I_{a,b}(𝖖)  =  Tr( ρ(L_a) · L_b )   satisfies   I_{a,b}(𝖖) = δ_{a,b} + O(𝖖) :

the canonical basis is **orthonormal to leading order in `𝖖`**. (The `𝖖⁰` term is
the `Δ = spin = 0` identity sector — see `docs/conjectures-step1-samples.md`.) For
a flavoured theory `I_{a,b} ∈ R((𝖖))` and the statement is on its identity (`χ₀`)
summand.

**How it is computed.** `BPSKAlgebra.trace` aggregates the central-direction
content as a residue of the Schur measure, exact in the `(1−𝖖^{2n})`-localized
ring and improvable to any q-order; `inner_product` evaluates the same exact
Schur formula along a single localized-ring path. The two-cutoff-stability shell
makes the trace **frame-sound** (independent of the presenting chart) and
**truncation-stable** — `trace(a, K)` agrees with `trace(a, K')` through `q^K`
for any `K' > K`, with no under-convergence warning.

## 3. That `S` factorises into palindromic BPS factors at all

`bps_factor_spectrum` (`docs/step4-BPSKAlgebra.md`) builds `S` as an
ordered product of one palindromic BPS factor `E^{(s)}_𝖖(X_γ)^{Ω(γ, s)}`
per pair `(γ, s)`, placed in any total order on those pairs and determined degree
by degree by `S`'s **leading data** — the `𝖖¹`-coefficient,
`−1` on the node charges and `0` elsewhere.

**What is conjectural is the surjectivity of that factorisation**: that every `S`
arising from a BPS quiver is reachable this way. The recursion itself is
deterministic and has no gate, so it always *returns* something; what is not a
theorem is that what it returns is the `S`. Two consequences are taken seriously
here rather than glossed:

- **The output is cross-checked, never assumed.** The self-test compares it
  against the independent Nahm-sum expansion of a known chamber spec
  (`Theory.S_from_spec`), against the retired peel recursion wherever that
  builds, and against the chart's own `[S|0⟩]_γ`. The peel recursion is kept
  reachable precisely so this comparison stays runnable.
- **The engine's own leading-data check is a regression guard, not evidence.**
  `verify_leading_data` is satisfied *by construction*, for any leading data —
  the recursion imposes it — so a green result there corroborates nothing about
  surjectivity. This is the enforced-versus-emergent distinction the project
  applies throughout: a verifier that construction forces is a guard against
  regressions, and citing it as support for a conjecture would overclaim.

A related, weaker statement concerns `factor_order_search`: that a *finite*
spin-0 factorisation exists at all is a property of the quiver and the order, not
a theorem. And the acceptance is **bounded rather than certain**, which is worth
stating precisely because it is easy to over-read.

The search screens on the factors **stopping** — ceasing to appear strictly below
the cutoff — and then *rebuilds* `S` and the candidate product a further
`confirm_extra` degrees and requires them to agree. That rebuild is the only
evidence there is, and it is finite: **a factorisation can agree with `S`
everywhere the check looks and disagree one degree past it**, and three
fixed-depth checks have each been defeated that way. No constant repairs this —
the counterexample simply lives past whatever depth was chosen, and sampling
cannot find the depth it did not reach.

So `Result.is_spec` is a claim **relative to `Result.confirmed_to`**, the degree
the run actually verified to, not a global statement that `S` factorises this
way; and where no spin-0 factorisation is found the search says so and returns
the sparsest order rather than a spurious one. This is the same
enforced-versus-emergent discipline applied to a *search*: report the depth
bought, not a certificate the cone cannot give.

## 4. `F_γ` and `S` from the relation alone — and what the relation does *not* fix

`FSBuilder` grows `F_γ` and `S` together out of `F_γ·S = X_γ + O(𝖖)` plus the
leading data, with **no `S` input and no support window**. Its agreement with the
F-finder is therefore genuine evidence for §1: the doubly-tropical interval that
bounds `F_γ`'s support is an *output* of this route, where for the F-finder it is
the enumeration window put in by hand.

The caveat is stated as an axiom-level fact rather than a footnote: **the
discovery relation by itself pins nothing.** Taking all multiplicities to
vanish gives `S = 1` and `F_γ = X_γ`, which satisfies `F_γ·S = X_γ + O(𝖖)`
exactly. The leading data is what makes the build determinate, and the self-test
exhibits the degenerate solution explicitly.

## Verification scope

What the tests actually certify (`tests/test_bps_flows.py`):

| check | scope |
|---|---|
| pentagon full axiom battery (`verify_canonical_basis`: unital / multiplicative / bar-invariant / orthonormality) + `KAlgebraIso` to the Step-1 `PentagonSampleKAlgebra` | 36 labels / 100 pairs, trace to q⁶ |
| flavoured hexagon trace | over `R((𝖖))` |
| node-deletion RG flow | certified against an independent UV realisation |
| atlas certificates | chart-invariance of multiply and the Schur index, monodromy = `ρ²`, Catalan chart counts for SU(2)-gauged `[A₁,Dₙ]` |
| the built `S` vs the Nahm-sum expansion of a known spec, and vs the chart's own `[S\|0⟩]_γ` | pentagon, cone degree 8 / 6 |
| order-independence of `S` (asserted non-vacuously: the orders must differ in BPS-factor count) | pentagon, every buildable placement order |
| the central charge selects the chamber (the one place `Z_γ` is in play) | pure SU(2): 2 strong-coupling factors vs the weak-coupling dyon tower + W boson (the one spin-1/2 state) |
| coverage where the peel gate trips | Markov (= N=2\*), cone degree 5 |
| the peel retirement | all five entry points refuse; both opt-ins reach the intact engine and it still agrees |
| `FSBuilder` vs the F-finder | 7 pentagon charges, positive and negative; support an output, not a window |
| the relation alone pins nothing | `Ω ≡ 0` ⇒ `S = 1`, `F = X_γ`, asserted |
| order search | a cored 3-cycle: 13 BPS factors collapse to 4 spin-0 ones, rebuilt through the independent Nahm-sum route; acyclic short-circuits to the node charges. `is_spec` is relative to `confirmed_to` — see §3 |
