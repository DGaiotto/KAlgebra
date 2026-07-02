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
to any q-order — no frozen-K cap, no sector that raises beyond a fixed order.)

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
The self-test `test_cones.py` runs **31 cone-contract cases** through the generic
`ConeKAlgebra` API, plus a `check_improvable` battery: trace-improvability probes
to high q-order for the closed-form families, and explicit-label batteries for
the realisations the generic cone loop does not reach (the ungauged polygons
`HexagonKAlg`/…/`DodecagonKAlg`, the ungauged `A1DevenKAlg`, the named
`A1D3/5/7ConeKAlg` cone presentations, and `SU2Nf2`/`SU2Nf3ConeKAlgebra`).

**Finite-type zoo** (the closed Argyres–Douglas / minimal theories) — Layer-1
reduction + the spine-free orthonormality bootstrap seeded by the exact Nahm-sum
`Tr(1)`:
`FinitePentagonKAlgebra` (A₂), `FiniteA3/A5/A7`, `FiniteA1D3…A1D8`,
`FiniteE6/E7/E8`, `FiniteHeptagonKAlgebra` (A₄).

**A1A_even — `A1A2kKAlg(k)`** (the reference family above): geometric cone-ray
chord labels + full M(2,2k+3) Andrews–Gordon character trace, every k.

**A1A_odd** (the (2k+4)-gons, k=1..4 = hexagon/octagon/decagon/dodecagon):
- *gauged* — `U1A1AoddKAlg(k)`: the U(1)-gauged family. Trace closes for **all
  k** — v-tower / long-chord (`a=2`) / diameter (`a=k+1`) seeds are the
  closed-form M(1,p) singlet characters (`u1_pgon_layer2`), and the k≥4
  intermediate chords (no closed form yet) are computed to arbitrary q-order by
  the spine-free orthonormality bootstrap. The multiply cross-products have no
  known closed form; the frozen tables `u1a1aodd_tables_k{k}.pkl` were computed
  with an RG-flow derivation not included in this repository (k is bounded by
  the provided tables).
- *ungauged* — `HexagonKAlg`/`OctagonKAlg`/`DecagonKAlg`/`DodecagonKAlg` =
  `[A₁,A_{2k+1}]`: the ungauged twins (centraliser of the gauge generator `E=μ`,
  measure-restored trace). Spine-free construct + multiply + arbitrary-q trace.

**A1D_odd** = `[A₁,D_{2k+3}]` = affine sl(2) at admissible level:
- `A1D3KAlg` — `[A₁,D₃]=[A₁,A₃]` (so(6)≅su(4)), the **explicit closed-form**
  sl(2)₋₄/₃ admissible characters (κ₀, κ₁^sym, κ₁^anti; Creutzig–Ridout) — a
  two-layer character trace, *not* a bootstrap.
- `FiniteA1D5` / `FiniteA1D7` — sl(2)₋₈/₅ / sl(2)₋₁₂/₇ via explicit closed-form
  admissible characters (`a1d5_layer2` / `a1d7_layer2`).
- `A1D3ConeKAlg` / `A1D5ConeKAlg` / `A1D7ConeKAlg` — the genuine D-type **cone**
  presentations: closed-form cone multiply off frozen inline Plücker tables +
  the arbitrary-q admissible-character trace (`a1dodd_layer2`). A1D7 is complete
  including the diameter seed (`a=3`); for k≥3 the trace raises rather than
  silently degrading (no known closed form).

**A1D_even** = `[A₁,D_{2k+2}]`, SU(2) (× U(1) when gauged) flavour:
- *gauged* — `U1A1DevenConeKAlgebra(1)` (D₄): closed-form multiply (frozen
  tables, oracle-free load), trace bootstrapped from `Tr(1)` alone (SU(2)
  per-irrep orthonormality sweep + all-orders monopole cyclicity); gauge sector
  the exact closed-form character, matter sector re-solvable to any q-order.
  **Only k = 1 is included**: the k ≥ 2 (D₆/D₈) spine-free matter bootstrap is
  not tractable at arbitrary order (its frozen tables would fail beyond
  K ≈ 8–12).
- *ungauged* — `A1DevenKAlg(k)` (D₄): the U(1) of the gauged D-even ungauged
  (centraliser of `X_{0,1}`; gauge charge → U(1) fugacity z; SU(2)×U(1)
  flavour). Trace reproduces `A1DevenRGKAlgebra` term-for-term.

**D₄ / SU(3)** — `SU3ADKAlg` = `[A₁,D₄]` = SU(3)₋₃/₂ with genuine **SU(3)
flavour** (coefficient ring `R(SU(3))`): `Tr_1` the closed-form Kac–Wakimoto
vacuum character of ŝl(3)₋₃/₂, `Tr_T`/`Tr_D` the orthonormality bootstrap seeded
by it. Layer-1 and the product multiply are carried in SU(3) Cartan fugacities
(weights, not characters; Weyl-symmetrised on the total), so non-self-dual
content (`T₀·T₂`'s `3+3̄`) is correct.

**E₇ (gauged)** — `U1E7ConeKAlgebra` = the u(1)-gauged E7 SCFT: a quantum-torus
cone (rank-1 gauge torus on `E=X_{(0,1)}`). Multiply loads frozen tables
(`u1e7_cone_tables.pkl`, computed with a derivation not included in this
repository); the magnetic sector vanishes and every neutral ray-word
is fixed by the E7 Nahm-sum vacuum + the forward-triangular orthonormality
bootstrap; ρ is the spine-free gauge-reflection.

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
(full Layer-2 for a3/hexagon) + `minimal_model_characters` (M(2,2k+3)); the
spine-free orthonormality bootstrap (`trace_uniqueness_proofs` + per-flavour
drivers); and `vacuum_nahm`, the exact Nahm-sum `Tr(1)` on the embedded BPS spec
(using only the spine-free `nahm_local`/`snf_kernel`/`qpoch`/`habiro`/`lattice`),
verified coefficient-for-coefficient against the RG-flow engine (`src/rg/`).

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
A1Deven→q³⁰, SU2Nf2 (Spin(4) index)→q¹². None of its computation paths uses a
realisation-engine module (the machine-checked `sys.modules` spine-freeness
assertions live in the Step-3 suites).

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

Two correct-but-slow notes: the U1A1Aodd(4) entry exercises the k≥4
intermediate-chord trace bootstrap (~30 s for one solve, cached per K); and the
8-node su2×u1 (a1d8) / 7-node u1 (e7) bootstrap seed-solves are heavy, so those
are exercised at the multiply/ρ level (their `Tr(1)` is fast and the trace math
is identical to a1d4/a1d6, which run the full battery).

## License

GPL-3.0-or-later (see `LICENSE`).
