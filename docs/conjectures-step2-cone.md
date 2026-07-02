# Conjectures — Step 2 (the cone realisations)

The cone layer (`src/cone/`) provides closed-form, spine-free `ConeKAlgebra`
realisations. The conjecture below is the one they directly bear on.

## Orthonormality of the canonical basis

**Conjecture.** The cone realisations presented in `src/cone/` satisfy the
`K_𝖖`-algebra axioms (`docs/axioms-and-bootstrap.md`); in particular, for the
canonical basis `{L_a}` the Schur pairing

    I_{a,b}(𝖖)  =  Tr( ρ(L_a) · L_b )

satisfies

    I_{a,b}(𝖖)  =  δ_{a,b}  +  O(𝖖) :

the canonical basis is **orthonormal to leading order in `𝖖`**.

### How the traces are computed here (no BPS/RG engine)

Each trace reduces (Layer 1, ρ²-cyclicity over the cone data) to a finite set of
elementary seeds, whose values come from **either** a known closed-form
character (`ad_characters` for a3/hexagon; the M(2,2k+3) Andrews–Gordon
characters in `minimal_model_characters` for A1A_even / `A1A2kKAlg`) **or** the
spine-free orthonormality bootstrap (`trace_uniqueness_proofs` + the per-flavour
drivers) seeded by the vacuum trace `Tr(1)`. `Tr(1)` is itself either a known
vacuum character or the exact Nahm sum on the BPS spec (`vacuum_nahm`). Every
path is exact and improvable to arbitrary q-order — there is no fixed-K cutoff
and no realisation engine.

## Verification scope

What the tests actually certify:

| check | scope |
|---|---|
| `tests/test_cones.py` — cone-contract cases (multiply / ρ / trace / orthonormality) | 31 cases; orthonormality at K = 3 on the unit plus up to 2 generators each |
| `tests/test_cones.py` — the `check_improvable` trace-improvability probes | traces extended to q³⁰–q⁷⁰ |
| `tests/test_sample_cone_iso.py` — sample ↔ cone isomorphisms | 3 isomorphisms, verified bidirectionally, with trace-equivariance to q¹² |
