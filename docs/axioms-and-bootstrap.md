# Axioms, verification, and the trace bootstrap

This note explains **what a `KAlgebra` is**, **how a concrete algebra is shown to
satisfy the axioms**, and **how strongly those axioms constrain the structure** —
the trace bootstrap being the prime example. For the formal statement see
[`conjectures-step1-samples.md`](conjectures-step1-samples.md); for the full
theory, Ambrosino–Gaiotto (DESY-25-035).

## The contract: six primitives

A concrete `A_𝖖[T]` subclasses `KAlgebra` (`src/core/kalgebra.py`) and supplies
six primitives:

| primitive | meaning |
|---|---|
| `coefficient_ring()` | the Z₊-ring `R` of coefficients (`Z`; `R(U(1)ⁿ)`; `R(SU(N))`; …) |
| `identity()` | the canonical label of the unit `1` |
| `multiply(a, b)` | structure constants in `Z[𝖖^±]` — the product `L_a · L_b` re-expressed in the canonical basis |
| `rho(a)` / `rho_inverse(a)` | the canonical automorphism ρ on labels, and its inverse |
| `trace(a, K)` | the ρ²-twisted trace `Tr(L_a)`, an `R`-power-series exact to `O(𝖖^K)` |

Flavoured algebras additionally implement a lift coordinate
(`r_label_decompose` / `r_label_compose`). Everything else — bilinear
`multiply_elements`, the Schur pairing `inner_product`, the Z-form ↔ R-form views,
`base_change` / `add_flavour`, and the `verify_*` checkers — is **derived** by the
base class.

## The axioms (and how they are checked)

On the canonical basis `{L_a}`, a `KAlgebra` must satisfy:

| axiom | checker |
|---|---|
| unit law `1·L_a = L_a = L_a·1` | `verify_unit_law(a)` |
| associativity `(L_a L_b) L_c = L_a (L_b L_c)` | `verify_associativity(a, b, c)` |
| ρ is an algebra automorphism `ρ(L_a L_b) = ρ(L_a) ρ(L_b)` | `verify_rho_is_automorphism(a, b)` |
| ρ⁻¹ inverts ρ | `verify_rho_inverse(a)` |
| ρ²-twisted cyclicity of the trace | `verify_trace_pairing_faces(a, b, K)` |
| **orthonormality** `I_{a,b} = Tr(ρ(L_a)·L_b) = δ_{a,b} + O(𝖖)` | `verify_orthonormality(a, b, K)` |

So "how the axioms are satisfied" is operational and reproducible: a concrete
algebra implements the six primitives, and a **green suite** (`run_tests.py`) is a
machine-checked certificate that the axioms hold on every tested label. All of it
is **exact** arithmetic — integer / `LaurentPoly` / `RElement` in 𝖖, with
truncation only inside the trace — so these are exact identities, never numerics.

## How the axioms constrain the structure: the trace bootstrap

Orthonormality is not a mild normalization; it is **rigid enough to determine the
traces**. This is the showcase of the release. The Step-2 `ConeKAlgebra`
realizations compute every trace **without any BPS / RG / quantum-torus engine**,
from the axioms plus a single seed.

`ConeKAlgebra` (`cone_kalgebra.py`) is a helper `KAlgebra` subclass that automates
this: it reduces the canonical (linear) basis to normal-ordered expressions in a
finite set of multiplicative **ray** generators, so a realization need only supply
its cone data — the reduction and trace are then derived.

The mechanism (`src/cone/`):

1. **Reduce (Layer 1).** Using only `multiply`, `ρ`, and ρ²-twisted cyclicity over
   the cone data, `Tr(L_a)` for *any* label reduces to a `Z[𝖖^±]`-combination of a
   **finite set of elementary seeds** — the vacuum `Tr(1)` and the single-generator
   traces. (`cone_kalgebra.py`, `cone_data.py`.)

2. **Pin the seeds (Layer 2).** Each seed is fixed in one of two ways:
   - a **known closed-form character** — e.g. the `M(2,2k+3)` Andrews–Gordon
     characters in `minimal_model_characters` for the `A1A2kKAlg` family, or
     `ad_characters`; **or**
   - the **orthonormality bootstrap** (`trace_uniqueness_proofs` + the per-flavour
     drivers): imposing `I_{a,b} = δ_{a,b} + O(𝖖)` supplies exactly enough linear
     relations to solve each remaining seed — *pinned from `Tr(1)` alone.*

3. **The one seed.** `Tr(1)` is the exact Nahm sum on the (frozen) BPS spectrum
   (`vacuum_nahm.py`) — a known vacuum character — exact to any 𝖖-order. It is a
   *seed for the bootstrap, not a truncated answer*: the Nahm sum itself runs to
   any `K`.

The entire trace of a cone algebra therefore follows from its `multiply` and `ρ`,
the single series `Tr(1)`, and the orthonormality axiom. Nothing is capped: every
path runs to arbitrary `𝖖^K`, and `test_cones.py` explicitly checks that traces
**improve past any fixed window** (no hidden cutoff). A green cone run — executed
with only this package on the import path — is thus a *proof* that the axioms
alone, not an external engine, fix the structure.

### Why this is the interesting part

The traces are the hard, theory-specific data (the Schur index). That they are
**over-determined by the axioms** — recoverable by bootstrap from `Tr(1)` — is
strong evidence for how tightly `A_𝖖[T]` is constrained. The cone package makes
that an executable demonstration; the hand-worked samples (`src/samples/`) show
the same rigidity on small cases.

## Two presentations, one algebra: `KAlgebraIso`

A further structural constraint: genuinely different presentations must define the
*same* algebra. `KAlgebraIso` (`src/core/kalgebra_iso.py`) certifies this via a
bijective label map, checking unit / round-trip / multiplicativity /
ρ-equivariance / trace-equivariance in **both** directions. `src/iso/` ships
verified witnesses that the by-hand samples and the cone realizations coincide:

| theory | sample (Step 1) | cone (Step 2) |
|---|---|---|
| pentagon `[A_1,A_2]` | `PentagonSampleKAlgebra` | `FinitePentagonKAlgebra` |
| A1A1 = SQED₁ | `SQED1SampleKAlgebra` | `U1SquareKAlg` |
| A1D2 = SQED₂ | `SQED2SampleKAlgebra` | `U1A1D2ConeKAlgebra` |

verified by `tests/test_sample_cone_iso.py`.

## Using a KAlgebra

With `sys.path` set (`conftest.py` and `run_tests.py` do this for you):

```python
from samples import PentagonSampleKAlgebra

A  = PentagonSampleKAlgebra()
L0 = (0, 1, 0)                                # generator L_0; labels are (i, a, b)

A.multiply(L0, L0)                            # L_0² in the canonical basis (an Element)
A.trace(L0, K=8)                              # Tr(L_0) to O(𝖖^8)
assert A.verify_orthonormality(L0, L0, K=6)   # I_{0,0} = 1 + O(𝖖)
```

To add your own algebra, subclass `KAlgebra` (a sample) or `ConeKAlgebra` (a cone
realization), implement the primitives, and add a case to the matching suite —
see [`../CLAUDE.md`](../CLAUDE.md).
