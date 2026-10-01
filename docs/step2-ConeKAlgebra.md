# ConeKAlgebra — closed-form, spine-free K_𝖖-algebras

**Step 2**, the cone layer of this repository (`src/cone/`): many examples of
K_𝖖-algebras `A_𝖖[T]` — U(1)-gauged Argyres–Douglas families, SU(2) gauge
theories, and the finite-type zoo — each realized with the **`ConeKAlgebra`
machinery**, a cone presentation of the **multiplicative generators** extending
the Step-1 `KAlgebra` contract. No realisation engine (no BPS-quiver or RG-flow
machinery) is used on any computation path; pure Python 3, no third-party
dependencies.

This layer is an **extension that depends on Step 1**, not a standalone copy: it
imports the core layer (`kalgebra`, `zplus_ring`, `laurent_poly`, `qpoch`,
`snf_kernel`, `sun_characters`, `tensor_zplus_ring`, `flavoured_kalgebra`) and
`KAlgebraIso` / the samples from `src/core/` and `src/samples/` by bare name —
none of those is duplicated here. `conftest.py` and `run_tests.py` put every
`src/<layer>/` directory on `sys.path`, so the imports resolve from the repo
root.

## What a `ConeKAlgebra` is

A `KAlgebra` whose canonical basis is organised into **cones** of multiplicative
generators (maximal q-commuting families of q-normal-ordered monomials). A
subclass supplies a `ConeData` instance (generators, q-commute cocycle,
cross-products, the cone↔canonical bijection); `multiply` is the generic
normal-ordering reduction over the cone data (words in the ray generators are
reduced back to canonical normal-ordered form), and `trace` is

```
trace(L) = Layer-1 reduction (ρ²-cyclicity over the cone data) → elementary seeds,
seed values  = known character expressions      (where they exist)
             | spine-free orthonormality bootstrap, seeded by Tr(1),
Tr(1)        = known vacuum character             (where it exists)
             | exact Nahm sum on the BPS spec     (vacuum_nahm — universal).
```

("Layer 1" is the reduction of an arbitrary trace to the elementary seeds by
`ρ²`-twisted cyclicity; "Layer 2" is the closed-form evaluation of those seeds.)

Every trace is **exact and arbitrarily q-improvable** — no fixed-K table, no
BPS/RG backend on the normal path. (Every class included here meets a strict
bar: *every* operation accepts arbitrary inputs and *every* trace is improvable
to any q-order — no frozen-K cap.  In the u(1)-gauged `[A₁,D_{2k+2}]` the
seeds come from closed forms and every other label is reduced onto them by
Layer 1, so no label reaches the exact transport that traced the products in
earlier releases; that transport, whose limit on the length of an intermediate
word is set from measured memory and raises past it rather than truncate, is
kept as the witness.)

### Cones — one class, three kinds

`cone_data.Cone` is a **single composable class** parameterised by a partition
of its mult-gens into *monomial* (identity change-of-basis), *quantum-torus*
(invertible directions `v·v⁻¹=1`), and *character* (SU(2) Chebyshev
`χ_k = U_k(χ_1)`) kinds, set via `Cone(parent, gens, torus_gens=…, char_gens=…)`
and queried with `is_monomial()` / `is_quantum_torus()` / `is_character()`. (The
former `MonomialCone`/`QTCone`/`CharacterCone` subclasses were folded into this
one class.)

## The reference family — A1A2k

`A1A2kKAlg(k)` is the reference cone implementation: **geometric cone-ray
labelling** of the chords of the (2k+3)-gon, and a **full closed-form Layer-2
trace** = the M(2,2k+3) Andrews–Gordon characters (`minimal_model_characters`).
It is closed-form and spine-free **for every k** (`A1A2kKAlg(1)` = pentagon =
M(2,5); `(2)` = heptagon = M(2,7); `(3)` = nonagon = M(2,9); `(k)` = M(2,2k+3);
…) — the whole A1A_even family, exact to any q-order with no engine. (The test
exercises a few sample k; there is no k restriction.)

## What's included

All realisations are spine-free (multiply + trace + orthonormality, no engine)
and their traces are arbitrarily q-improvable, organised by Dynkin family below.
For every ADE finite-type algebra — and its u(1)-gauged version where the flavour
has a u(1) — the class that serves it is self-contained: no frozen trace data and
no BPS or RG engine on any serving path, products and traces defined on every
label to any order.  For the A and D families the canonical-basis labels are
geometric: diagonals of a polygon, or curves of a once-punctured polygon.  The
self-test `test_cones.py` runs **35 cone-contract cases** through the generic
`ConeKAlgebra` API, a `check_improvable` battery (trace-improvability probes to
high q-order, and explicit-label batteries for the realisations the generic cone
loop does not reach), and `check_ade_rows` (the geometric labels, the complete
generator sets, the Z-form round trip, and the corrections listed below), and
ends by asserting that no module of `src/bps/` or `src/rg/` was imported.

**Finite-type zoo** (the closed Argyres–Douglas / minimal theories):
`FinitePentagonKAlgebra` (A₂), `FiniteA3/A5/A7`, `FiniteA1D3…A1D8`,
`FiniteE6/E7/E8`, `FiniteHeptagonKAlgebra` (A₄).  Each carries its cone table
(product data) and reduces a trace by Layer 1 to elementary seeds, which
`elem_traces` serves from closed forms — `a1d5` / `a1d7` from the sl(2)
admissible characters (`a1d5_layer2` / `a1d7_layer2`), `e6` / `e8` from W₃(3,7) /
W₃(3,8) character recipes (`w3_seeds`), `e7` from theta-product recipes and the
Bershadsky–Polyakov vacuum (`e7_seeds`) — or from a geometric class through a
generator map built and certified at runtime: `pentagon` / `heptagon` from
`A1A2kKAlg(1)` / `A1A2kKAlg(2)` (`aeven_seeds`), `a3` / `a5` / `a7` from
`ungauge_u1a1aodd(k)` (`aodd_seeds`), `a1d3` from `A1DoddConeKAlg(0)`
(`a1d3_seeds`), `a1d4` from `SU3ADKAlg` restricted to SU(2)×U(1)
(`a1d4_seeds`), `a1d6` / `a1d8` from `A1DevenKAlg(2)` / `A1DevenKAlg(3)`
(`a1deven_seeds`).  No frozen trace table remains (`elem_trace_data` is empty)
and no orthonormality bootstrap is on the serving path; the bootstraps stay as
witnesses of `elem_traces.generate`.  *Corrections carried by this layer:* the
frozen `a5` / `a7` tables of earlier releases were wrong near the top of their
windows (the `a5` flavour tails clipped from 𝖖¹⁵), and their su2u1 route peeled
the flavour slots of `a1d4` / `a1d6` / `a1d8` the wrong way round (no SU(2)
triplet at 𝖖² in the `a1d4` vacuum); both are pinned in `check_ade_rows`.
The `A` and `D` entries name their canonical basis geometrically:
`geometric_label(label)` is the multiset of curves — diagonals of the polygon,
or curves of the once-punctured polygon — that names the element in the family
class, read through the certified generator map that serves the entry's traces
(`zoo_geometry`); for `a1d5` / `a1d7`, whose traces stay on their closed forms,
the map onto `A1DoddConeKAlg(1)` / `A1DoddConeKAlg(2)` is found and certified at
runtime by `a1dodd_seeds`.  The `E` entries have no geometric labelling.

**A1A_even — `A1A2kKAlg(k)`** (the reference family above): geometric cone-ray
chord labels (`curve(x, ell)`: the diagonal from marked point `x` to `x + ell`
of the (2k+3)-gon, and `geometric_label(label)` a label's multiset of diagonals;
ρ is the rotation) + full M(2,2k+3) Andrews–Gordon character
trace, every k.

**A1A_odd** (the (2k+4)-gons; k=1..4 = hexagon/octagon/decagon/dodecagon, and
any k):
- *gauged* — `U1A1AoddKAlg(k)`: the U(1)-gauged family, at **every k** with no
  stored data.  The letters are the diagonals of the (2k+4)-gon plus the gauge
  letter `E^{±1}` (`geometric_label`); a letter's magnetic charge is set by the
  parities of its endpoints.  Products are the analytic peel (the even family's
  two arc rules on a rank-2 torus pairing); traces are Layer 1 plus **one**
  closed form for every chord seed, `u1_pgon_layer2.singlet_chord_trace` — a
  difference of two M(1,p) singlet module characters with a 𝖖-power prefactor,
  the gauged analogue of `A1A2kKAlg`'s minimal-model rule.  *Correction:* the
  fitted long-chord and diameter forms of earlier releases were wrong from about
  𝖖²⁶; earlier releases also shipped a frozen predecessor of this class
  (`u1a1aodd_tables_k{1..4}.pkl`), which capped k at 4.  `U1HexagonKAlg` keeps
  its own letters and computes through `U1A1AoddKAlg(1)`; its `geometric_label`
  names each letter by the same diagonal.
- *ungauged* — `[A₁,A_{2k+1}]`: `ungauge_u1a1aodd(k)` (any k), with the named
  `OctagonKAlg`/`DecagonKAlg`/`DodecagonKAlg` = `UngaugedPolygonKAlg(2, 3, 4)`
  and `HexagonKAlg` (over `U1HexagonKAlg`): the centraliser of the gauge letter
  `E`, `E` promoted to the flavour fugacity, measure-restored trace.  A label is
  a BALANCED multiset of non-crossing diagonals of the (2k+4)-gon — as many
  even–even as odd–odd diagonals — which is the geometric labelling of this
  family (`geometric_label`; `HexagonKAlg` reads it through
  `ungauge_u1a1aodd(1)`); `mult_generators()` returns the complete set, the
  mixed-parity diagonals and the non-crossing (even–even, odd–odd) pairs:
  6 / 24 / 65 / 144 at k = 1..4 (earlier releases listed only the single
  diagonals, 3 / 8 / 15 / 24).

**A1D_odd** = `[A₁,D_{2k+3}]` = affine sl(2) at admissible level:
- `A1D3KAlg` — `[A₁,D₃]=[A₁,A₃]` (so(6)≅su(4)), the **explicit closed-form**
  sl(2)₋₄/₃ admissible characters (κ₀, κ₁^sym, κ₁^anti; Creutzig–Ridout) — a
  two-layer character trace, *not* a bootstrap.
- `FiniteA1D5` / `FiniteA1D7` — sl(2)₋₈/₅ / sl(2)₋₁₂/₇ via explicit closed-form
  admissible characters (`a1d5_layer2` / `a1d7_layer2`).
- `A1D3ConeKAlg` / `A1D5ConeKAlg` / `A1D7ConeKAlg` (= `A1DoddConeKAlg(k)`,
  k = 0, 1, 2, and any k) — the genuine D-type **cone** presentations: closed-form
  cone multiply from the arc rules of the once-punctured polygon
  (`a1dodd_cone_data`, built for every k; the stored tables of earlier releases,
  `a1dodd_cone_tables`, are gone) + the arbitrary-q admissible-character trace
  (`a1dodd_layer2`, a closed-form recipe for every seed at every k).
  `geometric_label` reads a label `(word, κ)` as `A1DnKAlg(2k+3)`'s
  `(curves, κ)`, the curves of the once-punctured polygon (next item).
- `A1DnKAlg(n)`, n odd ≥ 3 — `[A₁,D_n]` on **geometric labels**: a label is
  `(curves, κ)`, a multiset of pairwise non-crossing curves of the n-gon with one
  interior puncture (`curve(x, ell, kappa=0)`: from marked point `x` to `x + ell`,
  `ell` the number of boundary edges on the side away from the puncture; `ell = n`
  is the loop around the puncture) and the SU(2) weight κ; ρ is the rotation.
  Products, ρ and trace go through `A1DoddConeKAlg((n−3)/2)` under the curve
  dictionary, certified as a `KAlgebraIso` (`a1dn_a1dodd_iso`).  Even n is
  refused, naming the even-D classes.  (Earlier releases shipped a different
  class under this name in `src/abe/`: the SU(2)-symmetrised quantum torus of the
  D_n chamber, whose trace was the flat-torus trace — `Tr 1 = 1 − 2𝖖² − 𝖖⁴` at
  n = 3, where `[A₁,D₃]` has `1 + χ₂𝖖² + …` — not the Schur index.)

**A1D_even** = `[A₁,D_{2k+2}]`, SU(2) × U(1) flavour, every k ≥ 1:
- *gauged* — `U1A1DevenConeKAlgebra(k)`, on **geometric labels** `(curves, e, κ)`:
  curves of the once-punctured (2k+2)-gon (`A1DnKAlg`'s convention), the power
  `e` of the gauge letter `E = X_{0,1}`, and the SU(2) weight κ in the label (the
  Z-form); accessors `curve(x, ell, e=0, kappa=0)` and `geometric_label`.  ρ is
  the rotation up to a power of `E`; which position carries that power is a
  documented convention (the full-turn argument in the class docstring).
  Products are closed forms (the arc rules of the curve frame,
  `u1a1deven_geometric_frame`).  Traces: the magnetic sector vanishes; the gauge
  sector is Creutzig's closed form (arbitrary q-order); a SEED — one curve of odd
  length, or a non-crossing pair of a charge +1 and a charge −1 curve, times a
  power of `E` — comes from its closed form (`u1a1deven_seed_characters`, any
  order; measured against the transport, not derived); every other label is
  reduced onto the seeds by the cone data's Layer-1 reduction
  (`ConeData.simplify_trace_via_cone_data`, on the `χ`-stripped labels), to any
  order; the pairing is multiply-then-trace.  (`seed_closed_forms=False` routes
  every trace with curves, and the pairing, through the exact transport of the
  class's closed-form RG image into `A1DoddConeKAlg(k−1) ⊗ QT(Z²)`, built
  without the flow (`u1a1deven_trace_transport`): the witness the closed forms
  and the reduction are checked against, whose limit on the length of an
  A1Dodd word raises rather than truncate.)
  (Earlier releases shipped only k = 1, as a ray-keyed table presentation loaded
  from two pickles; this class returned `RLaurent` coefficients inside a Z-form
  `Element`, so `to_R_form` raised.)
- *ungauged* — `A1DevenKAlg(k)`: the U(1) of the gauged class ungauged
  (centraliser of `E`; SU(2)×U(1) flavour).  Labels are the gauged labels
  `(F, e, κ)` in the centraliser, `F` a balanced multiset of curves (as many
  magnetic charges +1 as −1), `L_{(F,e,κ)} = z^{−e}·χ_κ·L_{(F,0,0)}` (Z-form);
  `mult_generators()` returns the complete set, the charge-0 curves and the
  non-crossing (+1, −1) pairs: 8 / 39 / 120 at k = 1 / 2 / 3 (earlier releases
  missed the pairs).  Trace reproduces `A1DevenRGKAlgebra` term-for-term on
  `Tr(1)`.

**D₄ / SU(3)** — `SU3ADKAlg` = `[A₁,D₄]` = SU(3)₋₃/₂ with genuine **SU(3)
flavour** (coefficient ring `R(SU(3))`).  Restricted to SU(2)×U(1) it is the
ungauged `A1DevenKAlg(1)`, and `geometric_label` names each canonical element
by that class's labels: `(curves, (p, q))`, `curves` a balanced multiset of
curves of the once-punctured square (`T_i` the loop at `i` with the curve
`(i + 1, 2)`, `D_i` the curve `(i + 1, 3)`), `(p, q)` the SU(3) weight.  The
three trace seeds come from the even-D k = 1 closed forms through that map,
summed over the gauge charge (`Tr_1` from Creutzig's gauge tower); the
Kac–Wakimoto vacuum character of ŝl(3)₋₃/₂ and the forward orthonormality pass
that served them in earlier releases are kept as their witnesses.  Product
traces ask their seeds for exactly the depth their reductions read (an earlier
padded depth could get the top order of a product trace wrong).  Layer-1 and
the product multiply are carried in SU(3) Cartan fugacities (weights, not
characters; Weyl-symmetrised on the total), so non-self-dual content
(`T₀·T₂`'s `3+3̄`) is correct.

**E₇ (gauged)** — `U1E7ConeKAlgebra` = the u(1)-gauged E7 SCFT: a quantum-torus
cone (rank-1 gauge torus on `E=X_{(0,1)}`). Multiply loads stored product tables
(`u1e7_cone_tables.pkl`, computed with a derivation not included in this
repository, on the gauged flow dressed with the central chord `(3, 0)`); the
magnetic sector vanishes and every magnetically neutral label, the `E`-tower
included, is traced through the ungauged `[A₁,E₇]` algebra's closed forms
(`FiniteE7KAlgebra`, by a label map certified by products); ρ is the stored
single-ray table plus the gauge reflection.  *Correction:* the tables of earlier
releases were learned from a flow dressed with the chord `(2, 2)`, which is the
u(1)-gauged `[A₁,D₇]` (`Tr(E^{±1}) = +𝖖²`, where the gauged `[A₁,E₇]` has
`−𝖖³`); the class refuses those tables on load.

**Pure / flavoured SU(2)**:
- `PureSU2KAlg` — pure SU(2) (`pure_su2_h_trace`, closed-form).
- `SU2Nf1KAlgebra` — SU(2)+N_f=1 (abelian flavour in the coefficients).
- `SU2Nf2ConeKAlgebra` — SU(2)+N_f=2, flavour **Spin(4)=SU(2)_L×SU(2)_R**
  (coefficient ring SU(2)⊗SU(2)): total multiply + total trace (the Spin(4)
  Schur index, arbitrary-q orthonormality bootstrap, no cap).
- `SU2Nf3ConeKAlgebra` — SU(2)+N_f=3, flavour **SU(4)** (matter SO(6)=Λ²4):
  total literal-word multiply (every magnetic level) + the SU(4) character-basis
  cyclicity+orthonormality bootstrap trace.

**SQED / U(1)-gauged AD cone standalones** (the A1A/A1D corners, each with a
certified Sample↔Cone `KAlgebraIso` to its Step-1 direct sample — see
`test_sample_cone_iso.py`):
- `U1SquareKAlg` — **A1A1** = SQED N_f=1 = U(1)-gauged `[A_1, A_1]`, unflavoured
  (a quantum-torus cone, `(m,n)` labels); trace = the SQED₁ Schur index,
  arbitrary-q.
- `U1A1D2ConeKAlgebra` — **A1D2** = SQED N_f=2 = U(1)-gauged `[A_1, D_2]` =
  `U_𝖖(𝔰𝔩₂)`, flavour **SU(2)** (the SU(2) spin carried in the label, as in
  `A1D3KAlg`); the `E·F = χ₁ + 𝖖K + 𝖖⁻¹K⁻¹` cone reproduces SQED₂'s
  `U_𝖖(𝔰𝔩₂)` straightener exactly.  trace = the SQED₂ index, arbitrary-q.

Trace machinery included: the closed-form characters `ad_characters`
(a3/hexagon), `minimal_model_characters` (M(2,2k+3)), `a1d5_layer2` /
`a1d7_layer2` / `a1dodd_layer2` (sl(2) admissible characters), `u1_pgon_layer2`
(M(1,p) singlet characters), `w3_seeds` and `e7_seeds` (the E-type recipes),
`exact_characters` (Creutzig's gauged D-even index) and
`u1a1deven_seed_characters` (the gauged D-even seeds); the generator-map
modules of the zoo (`aeven_seeds`, `aodd_seeds`, `a1d3_seeds`, `a1d4_seeds`,
`a1deven_seeds`, and `a1dodd_seeds`, which carries the geometric labels of
`a1d3` / `a1d5` / `a1d7`) and `zoo_geometry`, the zoo's geometric labels through
those maps; the spine-free orthonormality bootstraps
(`trace_uniqueness_proofs` + per-flavour drivers), kept as witnesses; and
`vacuum_nahm`, the exact Nahm-sum `Tr(1)` on the embedded BPS spec (using only
the spine-free `nahm_local`/`snf_kernel`/`qpoch`/`habiro`/`lattice`), verified
coefficient-for-coefficient against the RG-flow engine (`src/rg/`).

## Quick start

```python
# from the repo root, with the src/ layer dirs on sys.path (run_tests.py and
# conftest.py do this automatically; modules import by flat name):
import sys, pathlib
sys.path[:0] = [str(p) for p in pathlib.Path("src").rglob("*")
                if p.is_dir() and p.name != "__pycache__"]

from a1a2k_kalg import A1A2kKAlg              # the reference family
A = A1A2kKAlg(1)                              # pentagon, M(2,5)
A.trace(A.identity(), K=70)                   # exact to q^70 — no BPS, no cap

from finite_e8_kalg import FiniteE8KAlgebra
FiniteE8KAlgebra().verify_orthonormality((), (), K=6)

from su2_nf2_cone_standalone import SU2Nf2ConeKAlgebra
N2 = SU2Nf2ConeKAlgebra()                     # SU(2)+N_f=2, flavour Spin(4)
N2.trace(N2.identity(), K=12)                 # the Spin(4) Schur index
```

## Tests

```bash
python3 run_tests.py        # the full gate (all layers), from the repo root
```

`test_cones.py` runs the generic contract (multiply / ρ / trace / orthonormality)
on every included cone algebra, then a `check_improvable` battery that traces to
high q-order to witness arbitrary q-improvability spine-free — e.g.
pentagon→q⁷⁰, A1A2k(2)→q⁶⁰, U1A1Aodd→q⁴⁰, U1A1Deven(1)→q⁷⁰, A1D{3,5}ConeKAlg→q⁴⁰,
SU3AD `Tr_T`→q³⁰, the A1D7 diameter seed→q³⁰, the ungauged polygons→q³⁰⁻⁴⁰,
A1Deven→q³⁰, SU2Nf2 (Spin(4) index)→q¹² — then `check_ade_rows` (above),
`check_seed_routes` (the routes the closed-form seeds serve, each against its
witness: `SU3ADKAlg`'s geometric labels and seeds, the exact seed depth of its
product traces, the gauged D-even Layer-1 route against the transport),
`check_geometric_labels` (the geometric labels of the A and D classes, each
through its certified map: the named `PentagonKAlg` / `HeptagonKAlg` samples of
`src/abe/`, `A1A2kKAlg`, `U1HexagonKAlg` / `HexagonKAlg`, `A1DoddConeKAlg`
against the curves of `A1DnKAlg`, and every A / D entry of the zoo), and
finally asserts that no module of `src/bps/` or `src/rg/` was imported.

**Step-1↔Step-2 correspondence.** A separate test certifies the cone realisation
against the Step-1 sample:

```bash
python3 tests/test_sample_cone_iso.py        # certifies the Step-1 ↔ Step-2 isos
```

It builds a `KAlgebraIso` between each Step-1 sample and its cone twin and runs
the full `verify_all` battery (unit / round-trip / multiplicative on generators +
all pairs / ρ-equivariant / trace-equivariant), both directions — engine-free
(`KAlgebraIso` + the samples are imported from Step 1, not copied).  Three
correspondences are certified:

  * pentagon : `FinitePentagonKAlgebra` ↔ `PentagonSampleKAlgebra`
    (`K_𝖖([A_1, A_2])`, a non-trivial cyclic mult-gen map);
  * A1A1     : `U1SquareKAlg` ↔ `SQED1SampleKAlgebra`
    (SQED N_f=1 = U(1)-gauged `[A_1, A_1]`, identity on `(m, n)` labels);
  * A1D2     : `U1A1D2ConeKAlgebra` ↔ `SQED2SampleKAlgebra`
    (SQED N_f=2 = U(1)-gauged `[A_1, D_2]` = `U_𝖖(𝔰𝔩₂)`, SU(2)-flavoured; the
    relabeling bijection `(m, n, k) ↔ (gauge, k)`, with the SU(2) spin `k` in
    the label and the `E·F` cross-product carrying `χ_1` as an `RLaurent[SU(2)]`
    daughter).

Every catalogued algebra runs the full battery; the slowest entry is the zoo's
a1d8 (about 30 s), whose seeds come through `A1DevenKAlg(3)`.

## License

GPL-3.0-or-later (see `LICENSE`).
