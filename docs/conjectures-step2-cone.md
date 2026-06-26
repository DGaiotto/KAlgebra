# Conjectures

This package ships closed-form, spine-free `ConeKAlgebra` realisations. The
single conjecture they directly bear on — exactly what `test_cones.py` checks —
is the one below. See Ambrosino–Gaiotto (DESY-25-035) for the full theory.

## Orthonormality of the canonical basis

For the canonical basis `{L_a}` of a `K_𝖖`-algebra `A_𝖖[T]`, the Schur pairing

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

### Where it is checked here

- `KAlgebra.verify_orthonormality(a, b, K)` tests `I_{a,b}=δ_{a,b}+O(𝖖)` on a
  pair of canonical labels.
- `test_cones.py` runs it across every shipped cone algebra, and additionally
  verifies the traces extend spine-free past the old frozen window (arbitrary
  q-improvability).
