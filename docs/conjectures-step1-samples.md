# Conjectures — Step 1 (the sample algebras)

The core layer (`src/core/` + `src/samples/`) provides the `K_𝖖`-algebra
contract, the coefficient-ring layer, and a set of sample algebras. The
conjecture below is the one those structures directly bear on.

## Orthonormality of the canonical basis

**Conjecture.** The sample algebras presented in `src/samples/` satisfy the
`K_𝖖`-algebra axioms (`docs/axioms-and-bootstrap.md`); in particular, for the
canonical basis `{L_a}` the Schur pairing

    I_{a,b}(𝖖)  =  Tr( ρ(L_a) · L_b )

satisfies

    I_{a,b}(𝖖)  =  δ_{a,b}  +  O(𝖖) :

the canonical basis is **orthonormal to leading order in `𝖖`**.

*Superconformal interpretation.* The pairing counts states with weight
`𝖖^{Δ + spin}`; `Δ ≥ 0` by unitarity, and `spin ≥ 0` because only Spin(3)
highest weights contribute (SO(3) invariance), so the `𝖖⁰` term is the
`Δ = spin = 0` sector — suggesting `I_{a,b} = δ_{a,b} + O(𝖖)`. (By itself this
argument controls only the `𝖖⁰` coefficient; concluding that it equals
`δ_{a,b}` uses the additional input that the line defects `L_a` are simple and
pairwise distinct.)

## Verification scope

What the tests actually certify:

| check | scope |
|---|---|
| `KAlgebra.verify_orthonormality(a, b, K)` | a single pair of canonical labels: no negative-`𝖖` window, and the `𝖖⁰` coefficient equals `δ_{a,b}` in the identity sector of the coefficient ring |
| `tests/test_samples.py` — the full axiom battery (unit law, `ρ`-automorphism on all pairs, associativity on triples, `ρ²`-cyclicity faces, orthonormality on all label pairs) | 9–16 labels per algebra, 7 algebra instances (the quantum torus `Q_𝖖(Z²)`, the pentagon `K_𝖖([A_1,A_2])`, SQED₁ / SQED₂ / SQED_{N_f}, and `QuantumTorusKAlg(B)` with non-degenerate and flavoured `B`), at K = 4–5 |
