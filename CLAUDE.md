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
O(𝖖)`. Four layers of algebras sit over one shared contract:

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
- **rg** (`src/rg/`): `RGKAlgebra` and the catalogue of RG flows it presents.
  `RGKAlgebra` is a `KAlgebra` subclass that is *defined* by an RG flow to a
  graded auxiliary; its whole API (`RG`, `multiply`, `ρ`, `trace`) is **computed
  live** from the flow data via the exact bilinear `RG(a)·S_RG` trace pairing —
  spine-free (no BPS/RG realisation engine). It depends on core/samples and, for some
  auxiliaries, on cone; see `docs/step3-RGKAlgebra.md`.
- **bps** (`src/bps/`): `BPSKAlgebra` and the cluster-mutation `BPSAtlas`. A
  `KAlgebra` *realised* from a BPS quiver (a Dirac pairing + node charges): the
  canonical basis is discovered from the Kontsevich–Soibelman spectrum generator
  via `F·S = X_γ + O(𝖖)`, and the whole API follows. This is the realisation
  **spine** — the engine Steps 1–3 deliberately avoid. Adding it does not weaken
  their spine-free guarantee: no module in the earlier layers imports it at import
  time, and every Step-3 suite asserts no spine module is loaded (shared,
  filesystem-derived list in `tests/_spine.py`). See
  `docs/step4-BPSKAlgebra.md`.
- **iso** (`src/iso/`): `KAlgebraIso` witnesses identifying a sample with its cone
  realisation.

## Run the tests (the validation gate)

```bash
python3 run_tests.py
```

`test_samples`, `test_cones`, `test_sample_cone_iso`, the eight Step-3 RG
self-tests (`test_rg_flows`, `test_a1an_chain`, `test_dn_chain`, `test_e_type`,
`test_flavoured_fork`, `test_over_pure`, `test_su2_gauged_chain`, `test_wild`),
and the Step-4 BPS self-test (`test_bps_flows`, run last — it imports the spine)
must stay green. No third-party packages, nothing to install. `pytest` is not a
supported entry point and is refused loudly by `conftest.py` (it would skip
`test_cones.py` and `test_sample_cone_iso.py`, and importing the BPS suite at
collection time defeats the spine-freeness assertions in the Step-3 suites) —
use `python3 run_tests.py`.

## Layout & import model (read before moving files)

```
src/core/      kalgebra.py kalgebra_iso.py            the contract + iso witness
               zplus_ring.py laurent_poly.py          coefficient rings + exact 𝖖-arithmetic
               tensor_zplus_ring.py tensor_kalgebra.py snf_kernel.py qpoch.py sun_characters.py flavoured_kalgebra.py
src/samples/   samples.py quantum_torus_kalgebra.py uq_sl2_pbw.py
src/cone/      cone_kalgebra.py cone_data.py … + the realisation zoo   (120 .py + 8 .pkl)
src/rg/        rgkalgebra.py grading.py graded_rg_solver.py … + the flow zoo   (24 .py)
src/bps/       bps_kalgebra.py bps_quiver_tools.py bps_atlas.py … the realisation spine   (18 .py)
src/iso/       pentagon_/u1square_/u1a1d2_…_sample_cone_iso.py
tests/         test_samples.py test_cones.py test_sample_cone_iso.py + 8 RG-flow test_*.py + test_bps_flows.py
docs/          axioms-and-bootstrap.md  conjectures-*.md  step{1,2,3,4}-*.md
```

Modules import one another by **bare name** (`from kalgebra import …`), not by
package path. Every `src/<layer>/` directory is placed on `sys.path` by
`run_tests.py` (the gate), which globs the subdirectories of `src/`
(`conftest.py` exists only to refuse `pytest`). Consequently:

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
- **New RG flow:** subclass `RGKAlgebra` in `src/rg/`, supplying its flow data
  (`auxiliary`, `grading`, the `S_RG` components, `apex`); add a case to one of
  the `tests/test_*` RG-flow suites.
- **New BPS realisation:** instantiate `BPSKAlgebra` in `src/bps/` from a quiver
  (Dirac pairing + node charges, optional spectrum generator); add a case to
  `tests/test_bps_flows.py`.
- **New isomorphism witness:** add a builder to `src/iso/` and a case to
  `tests/test_sample_cone_iso.py`.
- **New flavour group:** add a `ZPlusRing` subclass in `src/core/zplus_ring.py`.

## Conventions

- The modules carry validated mathematics — extend by subclassing; do not rewrite
  the algebra core.
- No third-party dependencies (pure Python 3, nothing to install).
- No floating-point arithmetic in the algebra core.
