# A_q[T] — K-theoretic Coulomb-branch algebras

A contract-first, **dependency-free (pure Python 3)** implementation of the
K-theoretic Coulomb-branch algebras `A_q[T]` of 4d N=2 gauge theories. Nothing to install:

```bash
python3 run_tests.py        # or:  pytest
```

It is meant to let researchers (and their coding agents) **understand, use, and
extend** these algebras — and to document, executably, **how the defining axioms
are satisfied and how strongly they constrain the structure**. The trace
bootstrap is the prime example; see
[`docs/axioms-and-bootstrap.md`](docs/axioms-and-bootstrap.md).

## What this is

Each `A_q[T]` is an associative algebra over `Z[q^±]` with a distinguished
**canonical basis** `{L_a}`. The headline result, verified throughout, is
**orthonormality** of that basis under the ρ²-twisted trace:

```
I_{a,b}(q) = Tr(ρ(L_a)·L_b) = δ_{a,b} + O(q).
```

The code is layered over one shared contract:

| layer | dir | what it provides |
|---|---|---|
| **core**    | `src/core/`    | the abstract `KAlgebra` contract (six primitives + axiom verifiers), the Z₊-ring coefficient layer, exact `Z[q^±]` arithmetic, and the `KAlgebraIso` witness |
| **samples** | `src/samples/` | **Step 1** — sample algebras worked by hand: pentagon, SQED₁/₂/_Nf_, the Z² quantum torus, and `QuantumTorusKAlg(B)` over an arbitrary lattice `(Γ, B)` |
| **cone**    | `src/cone/`    | **Step 2** — `ConeKAlgebra` (a helper `KAlgebra` subclass that automates the multiplicative structure, reducing the canonical basis to normal-ordered expressions in multiplicative *ray* generators) plus a large catalogue of standalone realizations and the finite-type zoo |
| **iso**     | `src/iso/`     | `KAlgebraIso` witnesses certifying a Step-1 sample and a Step-2 cone realization are the *same abstract algebra* (pentagon, A1A1, A1D2) |

## The axioms constrain the structure — the trace bootstrap

Orthonormality is rigid enough to **determine the traces**. The Step-2
`ConeKAlgebra` realizations compute every structure constant and trace with **no
realisation engine** (no BPS-quiver / RG-flow / quantum-torus backend): ρ²-
cyclicity reduces any trace to a few elementary seeds, which orthonormality then
pins from the single vacuum seed `Tr(1)` (an exact Nahm sum). Every path is exact
and improvable to arbitrary q-order — no fixed-K cutoff — and the cone self-test
runs with only this code on the path, so a green run is itself the proof of
engine-freeness. Full account: [`docs/axioms-and-bootstrap.md`](docs/axioms-and-bootstrap.md).

## Using a KAlgebra

```python
from samples import PentagonSampleKAlgebra      # path set by conftest.py / run_tests.py

A  = PentagonSampleKAlgebra()
L0 = (0, 1, 0)                                  # the generator L_0; labels are (i, a, b)

A.multiply(L0, L0)                              # L_0² in the canonical basis
A.trace(L0, K=8)                                # Tr(L_0) to O(q^8)
assert A.verify_orthonormality(L0, L0, K=6)     # I_{0,0} = 1 + O(q)
```

## Running the tests

`python3 run_tests.py` runs the three contract suites and prints:

- `ALL SAMPLE CONTRACT TESTS PASSED`              (`tests/test_samples.py`)
- `ALL … CONE CONTRACT TESTS PASSED`              (`tests/test_cones.py`)
- the Step-1 ↔ Step-2 `KAlgebraIso` certification (`tests/test_sample_cone_iso.py`)

Pure Python 3, no third-party packages. The modules import one another by **bare
name**; `conftest.py` / `run_tests.py` put each `src/` layer on `sys.path`, so do
**not** nest the modules under a package directory or add `__init__.py`.

## Documentation

- [`docs/axioms-and-bootstrap.md`](docs/axioms-and-bootstrap.md) — the contract,
  how the axioms are checked, and how they constrain the structure (the trace bootstrap)
- `docs/conjectures-step1-samples.md`, `docs/conjectures-step2-cone.md` — the
  orthonormality conjecture and its cone-level form
- `docs/step1-KAlgebra.md`, `docs/step2-ConeKAlgebra.md` — per-layer notes
- `CLAUDE.md` — orientation for contributors and coding agents

## License

GPL-3.0-or-later — see `LICENSE`.
