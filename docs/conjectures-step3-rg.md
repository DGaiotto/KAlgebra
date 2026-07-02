# Conjectures — Step 3 (the RG flows)

The RG layer (`src/rg/`) provides the live RG-flow engine. Two statements from
the Schur-quantization RG framework are the ones it directly bears on — and, in
the constructive spirit of the project, the engine does not merely *check* them,
it *uses* them to **build** the algebra.

## 1. RG intertwining (the construction relation)

For an RG flow from a UV theory to an IR (auxiliary) theory with spectrum
generator `S_RG`, the RG map `RG` and the twisted automorphisms `ρ` satisfy

    RG(a)·S_RG  =  S_RG·ρ_IR⁻¹(RG(ρ_UV(a)))  =  L_{apex(a)} + O(𝖖) :

`RG(a)·S_RG` reproduces the IR apex label `L_{apex(a)}` of the UV label `a` to
leading order, and intertwines `ρ_UV`/`ρ_IR`. (`F·S = X_γ + O(𝖖)` is the leading
BPS special case.)

**Constructive use here.** `RGKAlgebra.RG(a)` is *solved* from the discovery
relation `RG(a)·S_RG = L_{apex(a)} + O(𝖖)` (`graded_rg_solver`, exact arithmetic
in the localization `Z[𝖖^±][(1−𝖖^{2n})^{−1}, n≥1]` over the grading cone). The
whole derived API (`multiply` via `from_ir_image`, `ρ`/`ρ⁻¹` via the mirror)
follows from this solve.

*Checked here:* `verify_rg_unital`, `verify_rg_multiplicative`,
`verify_rg_bar_invariant`, `verify_rho_is_automorphism`, and the certified
`KAlgebraIso` of each reference flow to its direct Step-1 sample.

## 2. Orthonormality of the canonical basis

**Conjecture.** The RG flows presented in `src/rg/` satisfy the `K_𝖖`-algebra
axioms; in particular, for the canonical basis `{L_a}` the Schur pairing

    I_{a,b}(𝖖)  =  Tr( ρ(L_a) · L_b )   satisfies   I_{a,b}(𝖖) = δ_{a,b} + O(𝖖) :

the canonical basis is **orthonormal to leading order in `𝖖`**. (The `𝖖⁰` term is
the `Δ = spin = 0` identity sector — see the Step-1 `docs/conjectures-step1-samples.md`.)

**How it is computed.** The trace pairing of an `RGKAlgebra` is the **bilinear
expansion** `I_{a,b} = Σ_{c,d} [RG(a)·S_RG]_c·[RG(b)·S_RG]_d·I^aux_{c,d}` over the
well-defined single-basis auxiliary pairing `I^aux_{c,d} = aux.inner_product(c,d)`
— *not* the ill-defined opposite-cone product `Tr_aux(ρ(S_RG)·…·S_RG)`. That
`I^aux` itself starts at `𝖖^0` (the IR's own orthonormality) is what lets the
exact-FS walk (the exact per-label evaluation of the `RG(a)·S_RG` products,
truncated to `𝖖^K` only at the end) skip beyond-`𝖖^K` contributions and stay
finite at each order.

## Verification scope

What the tests actually certify:

| check | scope |
|---|---|
| RG-unitality / RG-multiplicativity / bar-invariance / orthonormality | 21 flows, on 4–20 labels each, at K = 4–6 |
| pentagon multiply vs the direct Step-1 sample | all 400 products |
| vacuum traces vs the standalone cone algebras | matched to q⁶–q¹² |
| truncation stability + zero-warning discipline | every trace exercised |
