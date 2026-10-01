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

The bar has one licensed exception, and it is the guard the argument above
presupposes absent: **(★)**, the affine-Weyl residue cancellation — a line's
`𝖖`-difference operator must preserve the Neumann module `R(G)`. That is a
membership condition on the presentation, not a pairing within the span, so it
rejects exactly the off-span content orthonormality cannot see. The (★)-guarded
solve (`star_bubbling`, in Step 7) applies it, with bar-palindromicity and a box
certificate, before it returns anything; the general-gauge-group classes build
every canonical element with it, and the constructive routes are kept only as an
independent construction to compare against.

## 3. One abstract algebra, many certified presentations

An abstract `A_𝖖[T]` admits several presentations — a BPS-quiver chart, a cone, an
RG flow, the abelianized quantum torus — and a `KAlgebraIso` certifies that two of
them are the same object (a bijective canonical-label map intertwining `multiply`,
`ρ`, and the trace). The object layer (`KAlgebraObject`) holds many presentations
under one roof with a **path-independence (coherence)** certificate: the transition
isomorphisms compose consistently around every loop.

### The RG-intertwining relation (how matter is built)

The abelianized presentation of a **matter** theory is a pure-gauge `AbeKAlgebra`
combined with an `RGKAlgebra` flow that adjoins the matter (`GMatterOverPure`,
`GMatterOverMatter`, Step 7). The flow relation

    RG(a)·S_RG  =  S_RG·ρ_IR⁻¹(RG(ρ_UV(a)))  =  L_{apex(a)} + O(𝖖)

(see `docs/conjectures-step3-rg.md`) is *used* to build the flow, and the native
`(G, N)` algebra is certified `≅` the flow exactly when its matter irreps are
distinct.

## Verification scope

What the tests actually certify (`tests/test_abe_flows.py`):

| check | scope |
|---|---|
| the tier's contract + orthonormality on pure U(2) | `PureGAbeKAlgebra(u_n(2))`: derived `multiply`/`ρ` equal the class's own; bar, ρ-automorphism, `I_{a,b}=δ+O(𝖖)`, `certify_canonical` |
| matter | `roster('u2-nf1')`: `I_{a,b}=δ+O(𝖖)` on magnetic, electric and flavour labels |
| SU(2) N=2\* flow | `SU2N2StarRGKAlgebra` over `R(U(1))`, exact `F·S` available |
| object layer | `pure_u2_object` / `u2_nf1_object`: `abe↔cone` full battery, `abe↔bps` multiplicative, `verify_coherence`; `pure_un_bps_iso(2)` full battery |
| SU(2) family | native `PureSU2KAlgebra` orthonormality; `pure_su2_object` (abe / cone / bps / skein, pairwise batteries, coherence, two skein landmarks) / `su2_nf1_object` (with RG-flow legs to pure SU(2) and to the pentagon, coherence) / `su2_unf_object` (the `Spin(2N_f)` cone ↔ bps pair, coherence); `abelianized_su2_bps_iso` |
| SU(3) / U(3) | `PureGAbeKAlgebra(su_n(3))` orthonormality; `pure_un_bps_iso(3)` round trip |
| flow-typed object layer | `subquiver_flow_object` (F-oracle vs co-solver, bridged by an `RGKAlgebraIso`); `su2a1d3_rg_object` / `su2a1d4_rg_object`, incl. the Abe + RG object |
