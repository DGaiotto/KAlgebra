# Conjectures

**Step 3** ships the live RG-flow engine (`src/rg/`). Two statements from the
Schur-quantization RG framework are the ones it directly bears on — and, in the
constructive spirit of the project, the engine does not merely *check* them, it
*uses* them to **build** the algebra. (For the framework, see the reference at
the end.)

## 1. RG intertwining (the construction relation)

For an RG flow from a UV theory to an IR (auxiliary) theory with spectrum
generator `S_RG`, the RG map `RG` and the twisted automorphisms `ρ` satisfy

    RG(a)·S_RG  =  S_RG·ρ_IR⁻¹(RG(ρ_UV(a)))  =  a + O(𝖖) :

`RG(a)·S_RG` reproduces the UV element `a` to leading order, and intertwines
`ρ_UV`/`ρ_IR`. (`F·S = X_γ + O(𝖖)` is the leading BPS special case.)

**Constructive use here.** `RGKAlgebra.RG(a)` is *solved* from the discovery
relation `RG(a)·S_RG = L_{apex(a)} + O(𝖖)` (`graded_rg_solver`, exact Habiro
arithmetic over the grading cone). The whole derived API (`multiply` via
`from_ir_image`, `ρ`/`ρ⁻¹` via the mirror) follows from this solve.

*Checked here:* `verify_rg_unital`, `verify_rg_multiplicative`,
`verify_rg_bar_invariant`, `verify_rho_is_automorphism`, and the certified
`KAlgebraIso` of each reference flow to its direct Step-1 sample.

## 2. Orthonormality of the canonical basis

For the canonical basis `{L_a}`, the Schur pairing

    I_{a,b}(𝖖)  =  Tr( ρ(L_a) · L_b )   satisfies   I_{a,b}(𝖖) = δ_{a,b} + O(𝖖) :

the canonical basis is **orthonormal to leading order in `𝖖`**. (The `𝖖⁰` term is
the `Δ = spin = 0` identity sector — see the Step-1 `docs/conjectures-step1-samples.md`.)

**How it is computed.** The trace pairing of an `RGKAlgebra` is the **bilinear
expansion** `I_{a,b} = Σ_{c,d} [RG(a)·S_RG]_c·[RG(b)·S_RG]_d·I^aux_{c,d}` over the
well-defined single-basis auxiliary pairing `I^aux_{c,d} = aux.inner_product(c,d)`
— *not* the ill-defined opposite-cone product `Tr_aux(ρ(S_RG)·…·S_RG)`. That
`I^aux` itself starts at `𝖖^0` (the IR's own orthonormality) is what lets the
exact-FS walk skip beyond-`𝖖^K` contributions and stay finite at each order.

*Checked here:* `verify_orthonormality(a, b, K)` on each reference flow (no
negative-`𝖖` window; the `𝖖⁰` coefficient is `δ_{a,b}`), and the trace of each
flow matches the exact direct Step-1 sample, truncation-stable.

## Reference

F. Ambrosino, D. Gaiotto, *Renormalization Group Flow in Schur Quantization*,
JHEP **02** (2026) 057, [arXiv:2503.16685](https://arxiv.org/abs/2503.16685)
[hep-th], DOI:[10.1007/JHEP02(2026)057](https://doi.org/10.1007/JHEP02(2026)057).
