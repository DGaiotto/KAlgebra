# BPSKAlgebra — the BPS-quiver realisation engine for `A_𝖖[T]`

**Step 4**, the BPS layer of this repository (`src/bps/`). Given a BPS quiver
(an antisymmetric Dirac pairing + node charges) and, optionally, a chamber
spectrum generator, this layer realises the full `KAlgebra` contract —
`multiply`, `ρ`/`ρ⁻¹`, `trace`, `inner_product`, the axiom verifiers,
`to_R_form`, `base_change` — exact and improvable to any q-order. Pure Python 3,
no third-party dependencies.

## This layer is the *spine*

Steps 1–3 are deliberately **spine-free**: they compute structure constants and
Schur indices with no realisation engine (Step 2 from frozen cone reductions,
Step 3 from a live RG flow to a graded auxiliary). **Step 4 is the spine itself** —
the BPS realisation that *discovers* the canonical basis from an IR chart. The
spine-free guarantee therefore applies to Steps 1–3 only; this layer is where
the BPS engine lives. (Seven of the eight Step-3 self-tests assert that no spine
module is imported, so adding this layer does not weaken that guarantee — no
module in the earlier layers imports `src/bps/` at import time; a handful of
cone modules contain lazy, verify-only imports.)

## What `BPSKAlgebra` does

A single BPS chart determines `A_𝖖[T]`:

- **Discovery.** Each canonical-basis element `L_a` is the unique `F_γ` whose image
  in the auxiliary quantum torus satisfies `F_γ · S = X_γ + O(𝖖)` with
  bar-invariant coefficients (`S = ∏_i E_𝖖(X_{γ_i})` the Kontsevich–Soibelman
  spectrum generator, with `E_𝖖(x) = (−𝖖x; 𝖖²)_∞^{−1}`). The F-finder solves
  this exactly in the localization `Z[𝖖^±][(1−𝖖^{2n})^{−1}, n≥1]` over the
  doubly-tropical charge interval — the finite charge window `[γ₋(a), γ⁺(a)]`
  between a label's two tropical images, which bounds the support of `F_γ`.
- **Multiply.** `F` is an algebra map to the quantum torus, so
  `F(L_a · L_b) = F(L_a) · F(L_b)`: structure constants are read off the easy torus
  product and recognised back on the canonical basis.
- **ρ.** The closed-form piecewise-linear half-monodromy `σ` from the spectrum.
- **Trace / Schur index.** The central-direction residue of the Schur measure,
  exact and arbitrarily q-improvable; a two-cutoff-stability shell makes it
  frame-sound (independent of the presenting chart) and truncation-stable.
- **Spec-free.** The spectrum generator can be **built recursively from the quiver
  alone** (`build_S=True`) — no chamber spectrum required. The half-monodromy `σ`
  then also has an axiom-derived spec-free form (`spec_free_sigma="principled"`),
  for theories with no finite spectrum.
- **Flavour.** A degenerate pairing auto-extracts `Γ_f = ker(B)` via SNF; the
  coefficient ring becomes `AbelianZPlusRing(rank=f)` and flavour shifts ride in
  the labels (folded onto μ-monomials by `to_R_form`).
- **Charts & RG.** Cluster-mutating the quiver/spectrum gives *another*
  `BPSKAlgebra` for the **same** abstract algebra (an often-infinite family of
  presentations linked by explicit `KAlgebraIso`s), and node-deletion gives
  directional RG flows.

```python
from bps_kalgebra import BPSKAlgebra

# The pentagon = A_𝖖([A₁, A₂]) from its BPS (A₂) quiver.
B = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)])
B.multiply((1, 0), (0, 1))             # (q)·L_(1, 1)
B.inner_product((1, 0), (1, 0), K=6)   # 1 - q^2 + q^4 + q^6 + O(q^7)   (Schur index)
B.verify_canonical_basis(K=6)          # unital / multiplicative / bar-invariant / orthonormality

# Spec-free: build S recursively from the quiver alone.
Bf = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)], build_S=True)

# A flavoured theory: the hexagon (ker B = (1,1,1), one U(1) flavour).
H = BPSKAlgebra(pairing=[[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
                node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
H.coefficient_ring()                   # AbelianZPlusRing(rank=1)
H.trace((0, 0, 0), K=4)                # RPowerSeries over R((q))
```

## Layering — Step 4 builds on Steps 1, 2, 3 (additive; nothing duplicated)

This layer contains **only** the BPS-spine modules and imports the rest from the
earlier layers by flat name:

- the **Step-1 core** (`kalgebra`, `zplus_ring`, `laurent_poly`, `qpoch`,
  `snf_kernel`, `quantum_torus_kalgebra`, `kalgebra_iso`, `flavoured_kalgebra`,
  `sun_characters`, `tensor_zplus_ring`, and the `samples`);
- the **Step-2 cone layer** (`cone_data`, `pentagon_cone_data`, …) — the canonical
  *presentations* the BPS realisations are certified against, plus `nahm_local`;
- the **Step-3 RG engine** (`rgkalgebra`, `grading`, `graded_rg_solver`, `habiro`,
  `q_number_poly`, `lattice`).

`conftest.py` / `run_tests.py` put every `src/<layer>/` directory on `sys.path`,
so the flat imports resolve across layers without any `PYTHONPATH` wrangling.

## The modules (`src/bps/`)

Realisation core — `bps_kalgebra` (the `KAlgebra` realisation),
`bps_kalgebra_internals` (F-solve, Schur-index accumulator), `bps_quiver_tools`,
`recursive_spectrum` (the spec-free `S` builder); `nahm_local` (the Nahm-sum
locals) is shared with Step 2 and lives in `src/cone/`.

Charts & spectrum — `chart_graph` (lazy mutation graph), `spec_shortening`,
`spec_sigma`. RG flows — `rg_flow`, `directional_subquiver_rg` (node-deletion
flows + certification harness). Lattice helpers — `lattice_mutation`, `mutation`.
Isomorphism witnesses — `bpskalgebra_iso`, `bpskalgebra_kalgebra_iso`.

The **atlas layer** is the higher structure for exploring a theory's *cluster*
mutations: `bps_atlas` (`BPSAtlas` — an ensemble of `BPSKAlgebra` charts with
automated, certified `KAlgebraIso` transition maps across mutation chains),
`bps_chart_object` (the per-mutation `KAlgebraIso` witnesses + the `ρ²` rotation
monodromy), and `kalgebra_object` (the certified-presentations holder). A BPS-quiver
mutation is a cluster (Fomin–Zelevinsky) mutation, and the atlas certifies it
preserves the whole
`K_𝖖` structure (multiply, `ρ`, and the Schur index), with the rotation monodromy
closing to `ρ²`. `BPSAtlas.mutation_complete()` folds the whole quiver-mutation
orbit by chart-iso, so it closes for any *finite* theory regardless of spectrum
budget (the Argyres–Douglas zoo `[A₁,A₂]…[A₁,E₈]` closes).

Example galleries (self-contained — BPS-chart literals, no extra dependencies):
- `bps_atlas_examples` — the Argyres–Douglas zoo; every finite-type theory
  *completes* (its rotation cycle) and *mutation-completes* (the folded atlas).
- `bps_atlas_gauge_examples` — the gauge atlases. Here `mutation_complete` doubles
  as a **mutation-finiteness detector**: the SU(2) / A₁ class-S theories are
  mutation-finite (the fold closes — SU(2)-gauged `[A₁,Dₙ]` realises the Catalan
  numbers 5, 14, 42, 132, …), while SU(3) is mutation-infinite (the fold runs away;
  the chambers are mostly *wild*). Restricting to charts where S-finding works
  (`mutation_complete(keep=…)`, walling off the wild chambers) folds the infinite
  SU(3) orbit to a finite 2-chart atlas (the tame core).

## Self-test

`tests/test_bps_flows.py` runs the pentagon `A_𝖖([A₁, A₂])` from its BPS quiver
(spectrum and spec-free), the axiom battery + orthonormality + trace
truncation-stability, a `KAlgebraIso` to the Step-1 `PentagonSampleKAlgebra`, a
flavoured hexagon, a node-deletion RG flow certified against an independent UV
realisation, the `BPSAtlas` demonstration (the pentagon rotation chamber chain:
per-edge battery, Schur-index chart-invariance, and the `ρ² = monodromy` closure),
and, from the example galleries, the pentagon and `[A₁,A₃]` `mutation_complete`
folds, the SU(2)-gauged `[A₁,Dₙ]` (n = 3–6) mutation-finiteness / Catalan chart
counts, and the SU(3) mutation-infinite / wild-chamber probes.

## Tests

```bash
python3 run_tests.py        # the full gate (all four layers), from the repo root
```

`run_tests.py` runs the Step-1/2/3 contract tests followed by the Step-4 BPS
self-test. The BPS suite is run **last**: it imports the spine, and ordering it
after the Step-1/2/3 suites keeps their spine-freeness assertions valid in a
single-process run. `test_cones.py` (Step 2) and `test_bps_flows.py` (Step 4, the
atlas/gallery folds) are the slowest — allow a few minutes each.

## License

GPL-3.0-or-later (see `LICENSE`).
