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
may be supplied (the chamber spectrum) or **built recursively from the quiver
alone** (the spec-free `build_S=True` constructor). `ρ` on labels is the
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

## Verification scope

What the tests actually certify (`tests/test_bps_flows.py`):

| check | scope |
|---|---|
| pentagon full axiom battery (`verify_canonical_basis`: unital / multiplicative / bar-invariant / orthonormality) + `KAlgebraIso` to the Step-1 `PentagonSampleKAlgebra` | 36 labels / 100 pairs, trace to q⁶ |
| flavoured hexagon trace | over `R((𝖖))` |
| node-deletion RG flow | certified against an independent UV realisation |
| atlas certificates | chart-invariance of multiply and the Schur index, monodromy = `ρ²`, Catalan chart counts for SU(2)-gauged `[A₁,Dₙ]` |
