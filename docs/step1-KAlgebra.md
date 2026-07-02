# KAlgebra — the K_𝖖-algebra contract and sample algebras

A contract-first, dependency-free (pure Python 3) implementation of
**K_𝖖-algebras** `A_𝖖[T]` — the fusion algebras of rotation-equivariant BPS
line defects in 4d N=2 theories, Lagrangian gauge theories and non-Lagrangian
(Argyres–Douglas) theories alike.

This is **Step 1**, the core layer of this repository (`src/core/` +
`src/samples/`): the abstract `KAlgebra` contract, the coefficient-ring
("Z₊-ring") layer it is defined over, a set of worked **sample algebras**, the
**quantum torus over an arbitrary lattice Γ** (`QuantumTorusKAlg`, the
canonical realisation), and the **isomorphism-witness** machinery `KAlgebraIso`
(certifying that two presentations are the same abstract algebra). The heavier
*realisation engines* (the cone reductions, the RG-flow engine, the BPS-quiver
engine) live in the other layers of this repository — `src/cone/`, `src/rg/`,
`src/bps/` — and nothing in this layer depends on them.

## The contract — `kalgebra.py`

`KAlgebra` is an abstract base class for an associative algebra `A_𝖖[T]` over
`Z[q^±]` with a distinguished **canonical basis** `{L_a}`. A concrete algebra
supplies **six primitives**:

| primitive | meaning |
|---|---|
| `coefficient_ring()` | the Z₊-ring `R` (`TrivialZPlusRing()` if unflavoured) |
| `identity()` | the label of the unit |
| `multiply(a, b)` | structure constants `L_a·L_b = Σ_c C^c_{ab}(q)·L_c`, with `C ∈ Z[q^±]` |
| `rho(a)`, `rho_inverse(a)` | the canonical algebra automorphism `ρ` on labels |
| `trace(a, K)` | the trace / Schur index, an `R`-power-series to order `𝖖^K` |

plus an optional **flavour-lift coordinate** (`r_label_decompose` /
`r_label_compose`) for flavoured theories — the label of `L_a` factors as a
section (an unflavoured canonical) dressed by a flavour character.

From these the base class **derives** the bilinear extension
`multiply_elements`, the Schur pairing `inner_product`, the R-module view
(`to_R_form` / `from_R_form`), functorial coefficient-ring change
`base_change(phi)`, and a battery of **axiom verifiers**:

- bar-involution (the antimultiplicative `q ↦ q⁻¹` map fixing `{L_a}`),
- `ρ` is an algebra automorphism, and `ρ⁻¹` inverts it,
- `ρ²`-twisted trace cyclicity,
- canonical-basis **orthonormality** `Tr(ρ(L_a)·L_b) = δ_{a,b} + O(𝖖)`.

## The ring layer — `zplus_ring.py`, `laurent_poly.py`, …

Coefficient rings are **Z₊-rings** (Lusztig–Ostrik) — rings with a distinguished
`Z_{≥0}`-basis and a duality `⋆`:

- `TrivialZPlusRing()` — `Z` (the unflavoured case),
- `AbelianZPlusRing(rank=n)` — `R(U(1)ⁿ) = Z[μ_1^±,…,μ_n^±]`,
- `SU2ZPlusRing`, `SU3ZPlusRing`, `SUNZPlusRing(N)` — the representation rings
  `R(SU(N))` (Clebsch–Gordan / Littlewood–Richardson multiplication).

Exact arithmetic is carried by `LaurentPoly` (`Z[q^±]`), `RLaurent` (`R[q^±]`)
and `RPowerSeries` (`R((q))` truncated). Functorial flavour change uses
`RingHom`s (`augmentation_hom`, `restriction_hom`, …) through `base_change`.

`flavoured_kalgebra.py` and `tensor_zplus_ring.py` support the `add_flavour`
method and tensor coefficient rings.

## Samples — `samples.py`

Each sample is a **direct** `KAlgebra` subclass — it implements the six
primitives by hand (no engine), and is a reference example of "how to write a
`K_𝖖`-algebra":

| sample | theory |
|---|---|
| `Z2QTorusSampleKAlgebra()` | the quantum torus `Q_𝖖(Z²)` (the canonical example) |
| `PentagonSampleKAlgebra()` | `K_𝖖([A_1, A_2])` — A₂ Argyres–Douglas / Yang–Lee, M(2,5) |
| `SQED1SampleKAlgebra()` | SQED₁ = U(1) + 1 hyper (unflavoured) |
| `SQED2SampleKAlgebra()` | SQED₂ = `U_𝖖(𝔰𝔩₂)`, SU(2) flavour (PBW form) |
| `SQEDNfSampleKAlgebra(N_f)` | SQED_{N_f} = U(1) + N_f hypers, **SU(N_f) flavour** |

`SQEDNfSampleKAlgebra(1)` reproduces SQED₁ and `(2)` the `u_±/v` presentation of
SQED₂; `SQED1`/`SQED2` are kept as named samples.

## The quantum torus over Γ — `quantum_torus_kalgebra.py`

`QuantumTorusKAlg(B)` is the quantum torus **as a function of the lattice**
`Γ = Zⁿ` with an antisymmetric integer form `B`: the canonical basis is the
Γ-monomials `L_γ`, with `L_γ · L_δ = 𝖖^{⟨γ,δ⟩} L_{γ+δ}` and `ρ(γ) = −γ`. The
constructor splits `Γ` by `B` into a **gauge** part (the non-degenerate
quotient `Γ_g = Γ/ker B`) and a **flavour** part `Γ_f = ker B`; flavour lands in
the coefficient ring `R(U(1)^{rk Γ_f})` and shows up where the trace is read out
(`Tr(L_γ) ≠ 0` exactly when `γ ∈ Γ_f`). This is the general form of the
`Z2QTorusSampleKAlgebra` sample: `QuantumTorusKAlg([[0,1],[-1,0]])` *is* the
symplectic Z² torus.

## Isomorphism witnesses — `kalgebra_iso.py`

`KAlgebraIso(source, target, forward_label_map, inverse_label_map)` certifies
that two `KAlgebra` presentations are **the same abstract algebra**. It carries
both directions of a bijective label map, extends them 𝖖-linearly to elements,
and offers verifiers: `verify_unit`, `verify_round_trip`,
`verify_multiplicative`, `verify_rho_equivariant`. (Example in
`test_samples.py`: the `B`-preserving frame change `M = [[1,1],[0,1]]` as an
automorphism of the Z² torus.)

## Quick start

```python
# from the repo root, with the src/ layer dirs on sys.path (run_tests.py and
# conftest.py do this automatically; the modules import by flat name):
import sys, pathlib
sys.path[:0] = [str(p) for p in pathlib.Path("src").rglob("*")
                if p.is_dir() and p.name != "__pycache__"]

from samples import (
    Z2QTorusSampleKAlgebra, PentagonSampleKAlgebra, SQEDNfSampleKAlgebra,
)

# Quantum torus Q_q(Z^2):  X_g X_h = q^<g,h> X_{g+h}
A = Z2QTorusSampleKAlgebra()
A.multiply((1, 0), (0, 1))               # -> q * X_{(1,1)}
A.inner_product((1, 0), (1, 0), K=6)     # Schur pairing, an RPowerSeries

# SQED_{N_f}: U(1) + N_f hypers, SU(N_f) flavour
S = SQEDNfSampleKAlgebra(3)
S.multiply((1, 0, ()), (-1, 0, ()))      # u_+ u_- = sum_k q^k chi_{(1^k)} v^k
S.verify_orthonormality((0, 1, ()), (0, 1, ()), K=5)   # True

# Quantum torus over an arbitrary lattice (Γ, B)
from quantum_torus_kalgebra import QuantumTorusKAlg
Q = QuantumTorusKAlg([[0, 1], [-1, 0]])  # symplectic Z^2
Q.multiply((1, 0), (0, 1))               # -> q * L_{(1,1)}

# Certify two presentations are the same algebra
from kalgebra_iso import KAlgebraIso
from kalgebra import Element; from laurent_poly import LaurentPoly
one = LaurentPoly.one()
iso = KAlgebraIso(Q, Q,
    lambda l: Element({(l[0] + l[1], l[1]): one}),   # M=[[1,1],[0,1]]
    lambda l: Element({(l[0] - l[1], l[1]): one}))
iso.verify_multiplicative([(Element({(1,0): one}), Element({(0,1): one}))], [])  # True
```

## Exact arithmetic

Structure constants, traces, and Schur pairings are computed in **exact**
arithmetic in `q`; truncation to `O(𝖖^K)` happens only at the very end (the
orthonormality relation `Tr(ρ(a)·b) = δ + O(𝖖)` carries an `O(𝖖)` tail). Working
in truncated series from the start destroys the exact cancellations.

## Tests

```bash
python3 run_tests.py                 # the full gate, from the repo root
python3 tests/test_samples.py        # the Step-1 suite alone, from the repo root
```

`tests/test_samples.py` exercises the contract verifiers and the orthonormality
relation on every sample, on `QuantumTorusKAlg` (non-degenerate and flavoured
`Γ`), and the `KAlgebraIso` verifiers on a frame-change automorphism.

## License

GPL-3.0-or-later (see `LICENSE`).

## Conjecture

The central conjecture these structures bear on — **orthonormality of the
canonical basis**, `I_{a,b}(𝖖) = δ_{a,b} + O(𝖖)`, which the samples and
`test_samples.py` verify — is stated in `docs/conjectures-step1-samples.md`.

## Reference

Companion code to a paper in preparation.
