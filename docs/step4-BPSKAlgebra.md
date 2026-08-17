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
- **Spec-free.** The spectrum generator can be **built from the quiver alone**
  (`build_S=True`) — no chamber spectrum required; see "Building `S`" below. The
  half-monodromy `σ` then also has an axiom-derived spec-free form
  (`spec_free_sigma="principled"`), for theories with no finite spectrum.
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

# Spec-free: build S from the quiver alone (`bps_factor_spectrum`, by default).
Bf = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)], build_S=True)

# A flavoured theory: the hexagon (ker B = (1,1,1), one U(1) flavour).
H = BPSKAlgebra(pairing=[[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
                node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
H.coefficient_ring()                   # AbelianZPlusRing(rank=1)
H.trace((0, 0, 0), K=4)                # RPowerSeries over R((q))
```

## Building `S` — and building `F_γ` together with it

`S` need not come from a chamber spectrum. Two constructions of it live in this
layer, they share no mechanism, and that is what makes their agreement evidence
rather than a self-check — **but only one of them is active.**

**`bps_factor_spectrum`** is the default and the only active builder.
Instead of solving anything, it prescribes `S`'s **leading data** — the
`𝖖¹`-coefficient, which is `−1` on the node charges and `0` elsewhere — and reads
the palindromic multiplicities `Ω(γ, s)` off degree by degree, writing

```
S  =  ∏_{(γ, s)}  E^{(s)}_𝖖(X_γ)^{Ω(γ, s)} ,
```

one **BPS factor** per **pair** `(γ, s)` — the contribution of the BPS states of
charge `γ` and spin `s`, of which there are `Ω(γ, s)`. There is no gate and no
`F`-solve, so it builds at *any* BPS quiver, including the ones with no finite
chamber at all (Markov = N=2\*). Truncation is along the **cone**, never in `𝖖`: a negative
bracket shifts high-`𝖖` terms down into the visible window, so a `𝖖`-truncation
of `S` is unsound.

**What the order is, and what it is not.** The recursion needs exactly one thing:
a **total order on the pairs `(γ, s)`** — one position per BPS factor, each
placeable at an arbitrary position among the ones already placed. It does *not*
order by charge, and it does not need a central charge. Three parametrizations,
from most to least general:

| | what it orders by | |
|---|---|---|
| `piece_key=` | `(γ, 2s) → sortable` | the contract: any total order on the pairs |
| `order_key=` | `γ → sortable` | the coarse form — all of `γ`'s pieces at one position, in ascending spin |
| `phases=` | one complex number per node | the linear form — order by `arg Z_γ` |

Each is strictly narrower than the one above it, and the last is narrow enough to
matter: at rank 2 the phase order is monotone in slope, so a linear central
charge reaches two orders where free insertion finds dozens. Read `phases=` as
"the linear parametrization", never as what the algorithm requires.

Two things about it are easy to get wrong, and both are pinned by the self-test:

- **`S` is order-independent; its factorisation is not.** Where the BPS factors
  are placed changes the content `Ω` but not the element. So `Ω` is a chamber's
  BPS spectrum only once a **central charge** `Z_γ` is supplied and the order is
  its phase order — which is also the only situation in which the charges
  organise into *rays* at all. Then the central charge **chooses the chamber**:
  a generic one puts pure SU(2) in its two-state strong-coupling chamber, while
  the weak-coupling one gives the dyon tower plus the W boson — the same `S`.
- **No central charge is involved by default.** On an acyclic quiver the default
  order comes from the source/sink strip and gives `S = ∏_a E_𝖖(X_{γ_a})` on the
  node charges alone — the same chamber a good central charge selects, reached
  with no complex numbers anywhere. Where the strip leaves a core it determines nothing, and the default
  places each `(γ, s)` piece independently — which is licensed, since `⟨γ,γ⟩ = 0`
  makes the pieces at one `γ` commute. A central charge is used *only* when
  `phases=` is supplied.

**The peel engine** (`recursive_spectrum`) removes a node, solves
`F_γ·S_sub = X_γ + O(𝖖)`, and reattaches `E_𝖖(F_γ)`. It is **retired**: every
entry point raises `RetiredEngineError`. It is gated rather than deleted, and
three opt-ins (`allow_retired=True`, the `enable_retired_peel_engine()` context
manager, `PEEL_RETIRED = False`) reach it intact — because it is the only
*independent* construction of `S` here, and a cross-check you cannot run is not a
cross-check. Where it does build, the two agree exactly; where its monomial-ray
gate trips (a character-valued peel — matter), only `bps_factor_spectrum` builds.

**Searching the order** (`factor_order_search`). Since the factorisation depends
on the order and some orders factor `S` far more simply than others,
`find_simple_factorisation` runs a BFS over **BPS-factor insertion points**,
looking for an order whose factors are all spin-0 and which **stops populating**
inside the cone — what a consumer needs.

⚠ **What `is_spec` claims is bounded, and deliberately so.** Stopping inside the
cone is a *screen*, not a certificate: a factorisation can agree with `S`
everywhere the check looks and disagree one degree past it. Three fixed-depth
checks have been refuted exactly that way. So the result carries
`confirmed_to` — the degree the run actually rebuilt `S` and the candidate
product at and found them equal — and `is_spec` is a claim **relative to that
degree**, not a global one. A caller who needs more buys it with
`confirm_extra=`. The `S`-builder is never "done". On an acyclic quiver it short-circuits to the strip order, which already
attains the provable floor (at cone degree 1 the product is empty, so every node
carries a factor in every order). Where the strip leaves a core it earns its
keep: a dozen-plus factors carrying nonzero spin collapse to a handful of
spin-0 ones, and every reported factorisation is rebuilt through the independent
Nahm-sum route before it is returned. Where no spin-0 factorisation exists it
returns the sparsest order rather than nothing.

**Building `F_γ` and `S` together** (`fs_builder`). The F-finder above takes `S`
as given and enumerates the doubly-tropical window `[γ₋(a), γ⁺(a)]`. `FSBuilder`
takes neither: from `F_γ·S = X_γ + O(𝖖)` plus `S`'s leading data it grows both at
once, placing at each cone degree the BPS factors the leading data forces and
then reading off the palindromic `F`-coefficients the relation forces. The
doubly-tropical interval is therefore an **output**, which is exactly what makes
its agreement with the F-solver evidence rather than a restatement.

The two moves are one alphabet — `[n]_𝖖 = χ_{(n−1)/2}`, the F-solver's peel basis
and the multiplicity basis of the `S` side being the same `Z`-basis of the
palindromic Laurent polynomials — which is why the two recursions interleave at
all rather than merely running side by side.

⚠ **The relation alone pins nothing.** `Ω ≡ 0` gives `S = 1` and `F = X_γ`, which
satisfies `F·S = X_γ + O(𝖖)` exactly. What makes the build deterministic is `S`'s
leading data, and `omega_policy=` exposes the remaining choice rather than hiding
it. The self-test asserts this degenerate solution explicitly, because a builder
that only ever demonstrated its own success would be evidence of nothing.

```python
from bps_kalgebra import BPSKAlgebra
from fs_builder import FSBuilder
from factor_order_search import find_simple_factorisation

B = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)])
B.spectrum_generator_from_factors(cutoff=6)            # S, exact, cone-truncated
B.verify_spectrum_generator_from_factors(cutoff=6)     # == this chart's own [S|0⟩]_γ

b = FSBuilder([[0, 1], [-1, 0]], [(1, 0), (0, 1)], (1, 1), 6)
b.run(); b.F(), b.spectrum_generator()              # F_γ and S, grown together

find_simple_factorisation([[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
                          [(1, 0, 0), (0, 1, 0), (0, 0, 1)], 5).spec
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
`bps_kalgebra_internals` (F-solve, Schur-index accumulator), `bps_quiver_tools`;
`nahm_local` (the Nahm-sum locals) is shared with Step 2 and lives in `src/cone/`.

Building `S` — `bps_factor_spectrum` (leading data + one palindromic BPS factor per
`(γ, s)` pair; the default), `recursive_spectrum` (the retired peel recursion, plus the
engine-agnostic scaffolding both share — spec extraction, the auto-stabilising
cutoff search, and the axiom-derived spec-free `σ`), `factor_order_search` (the
search over placement orders for a finite spin-0 factorisation), and `fs_builder`
(`F_γ` and `S` grown together out of `F_γ·S = X_γ + O(𝖖)`).

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

It also covers the `S`-building layer above: `bps_factor_spectrum` against the
independent Nahm-sum expansion of a known chamber spec, its order-independence
(asserted non-vacuously — the orders must give *different* factor counts, or the
check would be testing nothing), the central charge selecting pure SU(2)'s
weak-coupling chamber with the W boson as its one spin-1/2 state, coverage of
Markov where the peel gate trips, the retirement closing every door while both
opt-ins still reach the intact engine, `FSBuilder` reproducing the F-solver over
a charge grid it never saw the support window for, and the order search
collapsing a cored quiver's factors to spin-0 ones that rebuild `S` independently.

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
