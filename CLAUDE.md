# K_𝖖-algebras — contributor & coding-agent orientation

Orientation for anyone (human or coding agent) working on this repository. For
the mathematics, read `README.md` and `docs/`; the central conceptual note is
[`docs/axioms-and-bootstrap.md`](docs/axioms-and-bootstrap.md).

## What this is

A dependency-free (pure Python 3) implementation of **K_𝖖-algebras** — the
axiomatisation of the fusion algebra of rotation-equivariant BPS line defects in
a 4d 𝒩=2 theory — and a range of examples. A K_𝖖-algebra is an algebra `A_𝖖`
over `Z[𝖖^±]` with a bar involution, a canonical basis `{L_a}`, an automorphism
`ρ`, and a `ρ²`-twisted trace under which `I_{a,b} = Tr(L_{ρ(a)} L_b) = δ_{a,b} +
O(𝖖)`. Two layers of algebras sit over one shared contract:

- **core** (`src/core/`): the abstract `KAlgebra` contract, the Z₊-ring
  coefficient layer (for flavoured algebras), exact `Z[𝖖^±]` arithmetic, and the
  `KAlgebraIso` isomorphism witness.
- **samples** (`src/samples/`): algebras written directly from the contract — the
  quantum torus, the pentagon `K_𝖖([A_1,A_2])`, and SQED₁/₂/_{N_f}_.
- **cone** (`src/cone/`): `ConeKAlgebra` and the catalogue of realisations it
  presents, plus the finite-type zoo. `ConeKAlgebra` is a `KAlgebra` subclass
  that **reduces the canonical basis to normal-ordered expressions in a finite
  set of multiplicative *ray* generators**, so a realisation supplies only its
  cone data; its traces are computed without a BPS/RG engine, by the bootstrap
  described in `docs/axioms-and-bootstrap.md`.
- **iso** (`src/iso/`): `KAlgebraIso` witnesses identifying a sample with its cone
  realisation.

## Run the tests (the validation gate)

```bash
python3 run_tests.py        # or:  pytest
```

`test_samples`, `test_cones`, and `test_sample_cone_iso` must stay green. No
third-party packages, nothing to install.

## Layout & import model (read before moving files)

```
src/core/      kalgebra.py kalgebra_iso.py            the contract + iso witness
               zplus_ring.py laurent_poly.py          coefficient rings + exact 𝖖-arithmetic
               tensor_zplus_ring.py snf_kernel.py qpoch.py sun_characters.py flavoured_kalgebra.py
src/samples/   samples.py quantum_torus_kalgebra.py uq_sl2_pbw.py
src/cone/      cone_kalgebra.py cone_data.py … + the realisation zoo   (118 .py + 8 .pkl)
src/iso/       pentagon_/u1square_/u1a1d2_…_sample_cone_iso.py
tests/         test_samples.py test_cones.py test_sample_cone_iso.py
docs/          axioms-and-bootstrap.md  conjectures-*.md  step{1,2}-*.md
```

Modules import one another by **bare name** (`from kalgebra import …`), not by
package path. Every `src/<layer>/` directory is placed on `sys.path` by
`conftest.py` (for `pytest`) and `run_tests.py` (for direct runs), each globbing
the subdirectories of `src/`. Consequently:

- do not add `__init__.py`, and do not rewrite imports to package-qualified form;
- a new module goes in the appropriate `src/<layer>/` directory, imported by its
  bare name;
- the `.pkl` tables in `src/cone/` are required data (frozen cone / ρ / trace
  tables), not optional.

## The contract

A concrete algebra subclasses `KAlgebra` and supplies six primitives —
`coefficient_ring`, `identity`, `multiply`, `rho` / `rho_inverse`, `trace` (and a
flavour-lift coordinate when flavoured). The base class derives the bilinear
product, the Schur pairing, the `Z`-form / `R`-form views, `base_change` /
`add_flavour`, and the `verify_*` axiom checkers (detailed in
`docs/axioms-and-bootstrap.md`). All arithmetic is exact in `𝖖`; truncation to
`O(𝖖^K)` happens only in the trace. The central relation is orthonormality of
the canonical basis, `I_{a,b} = δ_{a,b} + O(𝖖)`, which is rigid enough to
determine the traces from the single seed `Tr 1`.

## Extending

- **New sample:** subclass `KAlgebra` in `src/samples/`, implement the primitives,
  add a case to `tests/test_samples.py`.
- **New cone realisation:** subclass `ConeKAlgebra` in `src/cone/`, supplying its
  cone data; add a case to `tests/test_cones.py`.
- **New isomorphism witness:** add a builder to `src/iso/` and a case to
  `tests/test_sample_cone_iso.py`.
- **New flavour group:** add a `ZPlusRing` subclass in `src/core/zplus_ring.py`.

## Conventions

- The modules carry validated mathematics — extend by subclassing; do not rewrite
  the algebra core.
- No third-party dependencies (pure Python 3, nothing to install).
- No floating-point arithmetic in the algebra core.
