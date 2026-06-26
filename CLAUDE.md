# A_q[T] — contributor & coding-agent orientation

Quick start for anyone (human or coding agent) picking up this repo. For the math
read `README.md` and `docs/`; the key conceptual doc is
[`docs/axioms-and-bootstrap.md`](docs/axioms-and-bootstrap.md). Full theory:
Ambrosino–Gaiotto, *K-theoretic Coulomb branches and quantum algebras*
(DESY-25-035).

## What this is

A contract-first, dependency-free (pure Python 3) implementation of the
K-theoretic Coulomb-branch algebras `A_𝖖[T]` of 4d N=2 gauge theories — two layers
of algebras over one shared contract:

- **core** (`src/core/`): the abstract `KAlgebra` contract, the Z₊-ring
  coefficient layer, exact `Z[𝖖^±]` arithmetic, and the `KAlgebraIso` witness.
- **samples** (`src/samples/`): Step 1 — `KAlgebra`s worked by hand (pentagon,
  SQED₁/₂/_Nf_, the Z² quantum torus, `QuantumTorusKAlg(B)` over a lattice).
- **cone** (`src/cone/`): Step 2 — `ConeKAlgebra` plus a catalogue of standalone
  realizations and the finite-type zoo. `ConeKAlgebra` is a **helper `KAlgebra`
  subclass that automates the multiplicative structure**: it reduces the canonical
  (linear) basis to normal-ordered expressions in a finite set of multiplicative
  **ray** generators, so a realization supplies only its cone data. Traces are
  computed **spine-free** (bootstrapped from `Tr(1)`, no BPS/RG/quantum-torus
  engine); these standalone classes are often faster than the engine realizations
  and double as auxiliaries for them.
- **iso** (`src/iso/`): `KAlgebraIso` witnesses certifying a Step-1 sample and a
  Step-2 cone realization are the same abstract algebra.

## Run the tests (the validation gate)

```bash
python3 run_tests.py        # or:  pytest
```

Three suites must stay green: `test_samples`, `test_cones`,
`test_sample_cone_iso`. No third-party packages, nothing to install.

## Layout & import model (read before moving files)

```
src/core/      kalgebra.py kalgebra_iso.py            the contract + iso witness
               zplus_ring.py laurent_poly.py          coefficient rings + exact 𝖖-arithmetic
               tensor_zplus_ring.py snf_kernel.py qpoch.py sun_characters.py flavoured_kalgebra.py
src/samples/   samples.py quantum_torus_kalgebra.py uq_sl2_pbw.py     Step-1 worked algebras
src/cone/      cone_kalgebra.py cone_data.py … + the realization zoo  Step-2 (118 .py + 8 .pkl)
src/iso/       pentagon_/u1square_/u1a1d2_…_sample_cone_iso.py        sample ↔ cone witnesses
tests/         test_samples.py test_cones.py test_sample_cone_iso.py
docs/          axioms-and-bootstrap.md  conjectures-*.md  step{1,2}-*.md
```

**Critical:** modules import one another by **bare name** (`from kalgebra import …`,
`from cone_data import …`) — not by package path. Every `src/<layer>/` dir must be
on `sys.path`; that bootstrap lives in `conftest.py` (pytest) and `run_tests.py`
(direct), each globbing every subdir of `src/`. Consequently:

- **Do not add `__init__.py`** and **do not rewrite imports** to package-qualified form.
- A new module goes in the right `src/<layer>/` dir and is imported by its bare name.
- The `.pkl` tables in `src/cone/` are **required** data (frozen cone/ρ/trace tables), not optional.

## The math in one screen

- A concrete algebra subclasses `KAlgebra` and supplies six primitives:
  `coefficient_ring`, `identity`, `multiply(a,b)`, `rho`/`rho_inverse`,
  `trace(a,K)` (+ the lift pair if flavoured). The base class derives
  `multiply_elements`, the Schur pairing, the Z↔R views, `base_change`/
  `add_flavour`, and the `verify_*` axiom checkers.
- **Exact arithmetic — no floats, ever.** Integer / `LaurentPoly` / `RElement`
  arithmetic in 𝖖; truncation to `O(𝖖^K)` happens only in the trace.
- Headline result: orthonormality `Tr(ρ(L_a)·L_b) = δ_{a,b} + O(𝖖)`. The axioms are
  rigid enough to **bootstrap the traces** from the single seed `Tr(1)` — see
  [`docs/axioms-and-bootstrap.md`](docs/axioms-and-bootstrap.md).
- `KAlgebraIso` (`src/core/kalgebra_iso.py`) certifies two presentations define the
  same algebra via a bijective label map (`verify_unit / round_trip /
  multiplicative / rho_equivariant / trace_equivariant`); `src/iso/`
  cross-certifies samples against their cone twins.

## Extending

- **New sample:** subclass `KAlgebra` in `src/samples/`, implement the six
  primitives, add a case to `tests/test_samples.py`.
- **New cone realization:** subclass `ConeKAlgebra` in `src/cone/` supplying its
  cone data; add a case to `tests/test_cones.py`.
- **New iso witness:** add a builder to `src/iso/` and a case to
  `tests/test_sample_cone_iso.py` (the label bijection is rigidified by ρ — build
  it ray-to-ray and certify with `verify_*`).
- **New flavour group:** add a `ZPlusRing` subclass in `src/core/zplus_ring.py`.

## Conventions

- The modules carry **validated** mathematics — extend by subclassing; don't
  rewrite the algebra core.
- No third-party dependencies ("pure Python, nothing to install" is load-bearing).
- No floating-point arithmetic in the algebra core.
