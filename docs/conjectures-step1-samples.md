# Conjectures

This package ships the `K_𝖖`-algebra contract, the coefficient-ring layer, and a
set of sample algebras. The single conjecture below is the one those
structures directly bear on — it is exactly what the samples and
`test_samples.py` check.

## Orthonormality of the canonical basis

For the canonical basis `{L_a}` of a `K_𝖖`-algebra `A_𝖖[T]`, the Schur pairing

    I_{a,b}(𝖖)  =  Tr( ρ(L_a) · L_b )

satisfies

    I_{a,b}(𝖖)  =  δ_{a,b}  +  O(𝖖) :

the canonical basis is **orthonormal to leading order in `𝖖`**.

*Superconformal interpretation.* The pairing counts states with weight
`𝖖^{Δ + spin}`; `Δ ≥ 0` by unitarity, and `spin ≥ 0` because only Spin(3)
highest weights contribute (SO(3) invariance), so the `𝖖⁰` term is exactly the
`Δ = spin = 0` identity sector — forcing `I_{a,b} = δ_{a,b} + O(𝖖)`.

### Where it is checked here

- `KAlgebra.verify_orthonormality(a, b, K)` tests it on a pair of canonical
  labels (no negative-`𝖖` window, and the `𝖖⁰` coefficient equals `δ_{a,b}` in
  the identity sector of the coefficient ring).
- `test_samples.py` verifies it across every sample: the quantum torus
  `Q_𝖖(Z²)`, the pentagon `K_𝖖([A_1,A_2])`, SQED₁ / SQED₂ / SQED_{N_f}, and the
  general quantum torus `QuantumTorusKAlg(B)` over a lattice `(Γ, B)` — both
  non-degenerate and flavoured (degenerate `B`).
