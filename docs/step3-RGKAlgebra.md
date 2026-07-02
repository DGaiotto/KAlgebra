# RGKAlgebra — the live RG-flow engine for `A_𝖖[T]`

**Step 3**, the RG layer of this repository (`src/rg/`). Where Step 2
(`ConeKAlgebra`, `src/cone/`) provides *frozen* closed-form reductions, this
layer provides the **engine itself**: an
`RGKAlgebra` is *defined* by an RG flow to a graded auxiliary, and its entire
`KAlgebra` API — `RG`, `multiply`, `ρ`/`ρ⁻¹`, `trace`, `inner_product` — is
**computed live** from the flow data. Pure Python 3, no third-party dependencies;
every module imports by flat name.

**Coverage.** The flows included span the rank-1 Argyres–Douglas landscape both
ways — A-type (`A1Aeven→U1A1Aodd` chain; ungauged `A1Aodd`), D-type
(`A1D3Sqed2`/`U1A1DevenSqed` gauged chain; ungauged `A1Deven`), E-type (E₆, E₈,
gauged `U1A1E7`; ungauged E₇) — together with the SU(2) **Lagrangian gauge**
corner (SU(2)+N_f, SU(2)×SU(2) bifund, SU(2)ⁿ quiver), the **nested SU(2)-gauged
chain** (an `RGKAlgebra` whose auxiliary is itself an `RGKAlgebra`, composing down
to pure SU(2)), and "wild" formal flows that exercise the engine past any
realising theory. Eight clean-room self-tests, all spine-free. (Higher-rank SU(N)
gauge theories are not included.)

## The layering (depends on Step 1, and Step 2 for the cone auxiliaries)

This layer is an **extension of the core** (`src/core/`, Step 1) and imports it by
flat name (`kalgebra`, `zplus_ring`, `laurent_poly`, `tensor_kalgebra`,
`quantum_torus_kalgebra`, the `samples`, and `kalgebra_iso`). The A-type, D-type,
E-type, fork, over-pure and wild flows additionally use the cone auxiliaries from
Step 2 (`src/cone/`: e.g. `a1a2k_kalg`, `u1a1aodd_kalg`, `a1d3_kalg`,
`a1dodd_kalg`, `pure_su2_h_cone_data` + `pure_su2_h_trace_analytic`). Stages are a
**non-exclusive union** — Step 3 *depends on* the earlier layers and duplicates
nothing. The exact arithmetic in the localization `Z[𝖖^±][(1−𝖖^{2n})^{−1}, n≥1]`
(the `habiro` module — the name is historical) and the lattice / q-number helpers
(`lattice`, `q_number_poly`) are shared with Step 2 and live in `src/cone/`; they
are imported by flat name, not copied into `src/rg/`.

`conftest.py` (for `pytest`) and `run_tests.py` put every `src/<layer>/` directory
on `sys.path`, so the flat imports resolve across layers without any
`PYTHONPATH` wrangling.

## What an `RGKAlgebra` is

A concrete flow supplies only its **defining data**:

- `auxiliary()` — the IR `KAlgebra` (the flow target);
- `grading()` — a `Γ_RG`-grading of the auxiliary (`grading.py`): a charge
  `deg(L) ∈ Γ_RG` on each auxiliary label, **additive** under `multiply`, plus a
  height functional `h` (an integral proxy for the physical central charge
  `Im Z_γ`) strictly positive on the appearing charges (⇒ a pointed cone);
- the spectrum generator `S_RG`, in two contracts — `_s_rg_component(p)` (the
  *exact* graded component `[S_RG]_p`) and `rg_generator(cutoff)` (the q-order
  window);
- `apex(a)` — the tropical identification of a UV label with its IR apex.

Everything else is **derived generically** (`rgkalgebra.py`):

- `RG(a)` is **solved** from the discovery relation `RG(a)·S_RG = L_{apex(a)} +
  O(𝖖)` (`graded_rg_solver.py`, exact arithmetic in the `(1−𝖖^{2n})`-localized
  ring) — not hand-coded;
- `multiply(a,b) = from_ir_image(RG(a)·_aux RG(b))`;
- `ρ`/`ρ⁻¹` from the same solve / its mirror;
- `trace` / `inner_product` by the **bilinear pairing** (see below).

## Computing the trace right (the engine-level rule)

The trace pairing is the **bilinear expansion**

    I_{a,b} = Σ_{c,d} [RG(a)·S_RG]_c · [RG(b)·S_RG]_d · I^aux_{c,d},   Tr(a) = I(1,a),

with `I^aux_{c,d} = aux.inner_product(c,d)` a *well-defined single-basis* pairing.
One must **never** form `Tr_aux(ρ(S_RG)·…·S_RG)`: `ρ(S_RG)` is a formal sum over
the negative grading cone and `S_RG` over the positive cone, so their product is
**not a well-defined auxiliary element** (only an early `𝖖^K` truncation makes it
finite, and the answer then marches with the cutoff). Bilinearity moves `ρ` and
the trace onto the well-defined `I^aux_{c,d}`, leaving a finite sum at each
`𝖖`-order. The per-charge components `[RG(a)·S_RG]_c` are computed **exactly per
output label** by an order-guided support walk (`_rg_times_s_rg_exact`): the walk
generates neighbours by multiplying a label by the `S_RG` ray-units in the
auxiliary (so it works for torus, non-torus, and nested/flavoured labels alike),
each component is fetched whole and expanded to `𝖖^K` **last** — exact to any `K`.

## The three reference flows

| flow | theory | auxiliary | grading `deg` | `S_RG` | corner |
|---|---|---|---|---|---|
| `U1SquareRGKAlgebra` | SQED₁ = U1Square | Z² quantum torus (`Z2QTorusSampleKAlgebra`) | `b` (a coordinate) | `E_𝖖(X_{0,1})` | the **torus** corner (`deg = id`) |
| `PentagonSquareSampleRGKAlgebra` | pentagon `K_𝖖([A₁,A₂])` (Yang–Lee) | SQED₁ (`SQED1SampleKAlgebra`) | `m` (magnetic charge) | `E_𝖖(u₋)` | the **non-torus** corner (`u₊u₋ = 1+𝖖v`) |
| `SQEDNfRGKAlgebra(N_f)` | SQED_{N_f} (U(1)+N_f hypers, SU(N_f) flavour) | Z² QT `.add_flavour(SUNZPlusRing(N_f))` | `b` | `∏_i E_𝖖(μ_i X_{0,1}) → χ_{SU(N_f)}` | the **flavoured / nested-aux** corner |

(Throughout, `E_𝖖(x) = (−𝖖x; 𝖖²)_∞^{−1}` is the quantum-dilogarithm factor from
which every `S_RG` is built.)

All three are **pure** `RGKAlgebra`s — no operation is overridden to a closed
form; `RG` is solved by the engine, not hard-coded. Each is certified by a
`KAlgebraIso` to the direct Step-1 sample (`SQED1SampleKAlgebra` /
`PentagonSampleKAlgebra` / `SQEDNfSampleKAlgebra`), with `multiply` and `trace`
matching exactly and the trace truncation-stable. The pentagon flow is the
non-torus case: its IR is a gauge theory, so `RG(a)·S_RG` spreads into `v`-bands
that the engine's multiply-based support walk discovers automatically. `SQED_{N_f}`
is the **flavoured** case: the auxiliary has nested `((a,b), χ)` labels and the
flavour fuses by Clebsch–Gordan — the trace is the nested-aux exact-FS bilinear
pairing ("exact-FS" = the exact per-label evaluation of the `RG(a)·S_RG`
products, truncated to `𝖖^K` only at the very end) and the multiply takes the
engine's non-additive-flavour branch.
`SQEDNfRGKAlgebra(1)` = SQED₁, `SQEDNfRGKAlgebra(2)` = SQED₂ (the SU(N_f)-character
content of `∏_i E_𝖖(μ_i x)` is the general-purpose `sunf_dilog` tool).

## The A-type Argyres–Douglas chain (`A1Aeven / U1A1Aodd`)

Beyond the three single-flow references, Step 3 includes a **two-leg RG chain**
walking the A-type AD theories down through the gauged-odd theory. Both legs are
**pure** exact-FS `RGKAlgebra`s (RG solved, no override), with spine-free
auxiliaries drawn from Steps 1 + 2:

| leg | flow | auxiliary | `S_RG` |
|---|---|---|---|
| 1 | `A1AevenToU1AoddRGKAlgebra(k)` — `[A₁,A_{2k+2}] → u(1)-gauged [A₁,A_{2k+1}]` | `U1A1AoddKAlg(k)` (Step 2) | `E_𝖖(X_L)` (magnetic tower) |
| 2 | `U1A1AoddToEvenQTRGKAlgebra(k)` — `u(1)-gauged [A₁,A_{2k+1}] → [A₁,A_{2k}] ⊗ QT[Z²]` | `A1A2kKAlg(k)` (Step 2) ⊗ `QuantumTorusKAlg` (Step 1) | `E_𝖖(X_{0,1}·L)` (short-chord tower) |

Leg 1's vacuum reproduces the standalone even `A1A2kKAlg(k+1)` Schur index; leg
2's reproduces the gauged-odd `U1A1AoddKAlg(k)` index exactly to q¹² (k=1 carries
a deep `q¹⁰` term that a windowed-heuristic trace would truncate — caught by the
exact-FS bilinear pairing). Leg 2's `S_RG = E_𝖖(X_{0,1}·L)` is a single
short-chord tower built natively over `TensorKAlgebra(A1A2kKAlg(k), QT_2D)`; the
generic engine solves `RG`, so no per-`k` hard-coded table is needed. Self-test:
`tests/test_a1an_chain.py` (also asserts no spine module is imported).

## The D-type gauged chain (`A1Dodd / U1A1Deven`)

The D-type analogue of the A-type chain — the gauged rungs of the A1Dodd–U1A1Deven
ladder. The SU(2) flavour is *intrinsic* throughout (it lives on the survivor /
SQED₂'s electric hypers), so each flow drops a **single** state, not a doublet and
no `add_flavour` spectator:

| rung | flow | auxiliary | `S_RG` |
|---|---|---|---|
| odd (D₃) | `A1D3Sqed2RGKAlgebra()` — `[A₁,D₃] → SQED₂` | `SQEDNfSampleKAlgebra(2)` (Step 1) | `E_𝖖(u₊)` (single SU(2)-singlet monopole) |
| even (D_{2k+2}) | `U1A1DevenSqedRGKAlgebra(k)` — `u(1)-gauged [A₁,D_{2k+2}] → A1Dodd(k-1) ⊗ QT[Z²]` | `A1DoddConeKAlg(k-1)` (Step 2) ⊗ `QuantumTorusKAlg` (Step 1) | `E_𝖖(X_{(0,1)}·L)` (single gauge-dressed doublet) |

Both pure exact-FS. A1D3Sqed2's vacuum reproduces the standalone `A1D3KAlg`
Schur index exactly (q² = the SU(2) adjoint χ₂); U1A1DevenSqed's vacuum is the
gauged-`[A₁,D₄]` index: the SU(2)-refined series is `1 + (χ₂ − 1)q² + …` (the
SU(2) current minus the gauged-U(1) subtraction), whose identity-character
summand is `1 − q² + …`. Self-test: `tests/test_dn_chain.py`. (The A1D3→SQED₂
single-**singlet** drop is the converse of the SQED₁⊗SU(2) doublet drop — same UV
theory, two distinct IR flows.)

## The exceptional E-type flows (`[A₁, E₆]`, `[A₁, E₈]`)

The flavourless exceptional AD theories are realised as the E-series analogues of the
A-chain's leg 1 — a single-node drop with `S_RG = E_𝖖(L)`, but `L` the **central
diameter** chord (vs the A-type's end short chord; *which chord is dropped* is the
whole A/E distinction):

| flow | theory | auxiliary | `S_RG` |
|---|---|---|---|
| `E6RGKAlgebra()` | `[A₁, E₆]` | `U1A1AoddKAlg(2)` = u(1)-gauged `[A₁,A₅]` (Step 2) | `E_𝖖(L_diam)` |
| `E8RGKAlgebra()` | `[A₁, E₈]` | `U1A1AoddKAlg(3)` = u(1)-gauged `[A₁,A₇]` (Step 2) | `E_𝖖(L_diam)` |
| `U1A1E7RGKAlgebra()` | u(1)-gauged `[A₁, E₇]` | `A1A2kKAlg(3) ⊗ QT(Z²)` (Step 2 ⊗ Step 1) | `E_𝖖(X_{(0,1)}·L_{(2,2)})` |

All three are pure exact-FS `RGKAlgebra`s (RG solved, no override) with spine-free
auxiliaries. E₆/E₈ are flavourless (the gauged-odd cone directly, no
`add_flavour`): their vacua have **no q² term** (as required); computed with this
engine to q¹⁰, they agree to q⁸ and diverge at q¹⁰ (E₆: 3, E₈: 4 — distinct
theories), while the self-test pins the series to q⁶. E₇ is presented via its
**u(1)-gauged** form: the ungauged `[A₁,E₇] → [A₁,A₆] ⊕ U(1)` carries the U(1) as
an `add_flavour(1)` spectator with a slow refined trace, so gauging the U(1) (one
leg of `QT(Z²)`, dressing the interior node-4 chord) puts it on the same exact-FS
engine — vacuum `1 − q² + q⁶ + … ` (q² = −1, the gauged-U(1) subtraction).
Self-test: `tests/test_e_type.py`. (E₇ is *also* presented in its **ungauged**
U(1)-flavoured form below — both presentations are pure exact-FS.)

## The ungauged `add_flavour` fork (`A1Aodd`, `E₇`, `A1Deven`)

The `add_flavour(…)` companions to the gauged forks: the dropped matter is kept as
a **spectator flavour** (not gauged), so the auxiliary is
`A1A2kKAlg(k).add_flavour(flavour)` and the trace is flavour-valued.

| flow | theory | flavour | `S_RG` |
|---|---|---|---|
| `A1AoddToEvenRGKAlgebra(k)` | `[A₁, A_{2k+1}] → [A₁, A_{2k}] ⊕ U(1)` | U(1) | `E_𝖖(μ·L)`, **end** chord |
| `E7RGKAlgebra()` | ungauged `[A₁, E₇] → [A₁, A₆] ⊕ U(1)` | U(1) | `E_𝖖(μ·L)`, **central** chord |
| `A1DevenRGKAlgebra(k)` | `[A₁, D_{2k+2}] → [A₁, A_{2k}] ⊕ U(2)` | U(2) | `E_𝖖(μ₁L)E_𝖖(μ₂L)`, two-fork doublet |

For the U(1) pair, *which* chord is dressed (end vs central) is the entire A/E
distinction — the exact analogue of the gauged fork (end → A-even, central →
E₆/E₈). A1Deven drops the **two-fork doublet** instead, giving a **U(2)** flavour
(the SU(2)×U(1) on the two dropped fork hypers; the U(2) ⊂ the full even-D
flavour). All are pure exact-FS (RG solved, no override): the trace is the generic
bilinear exact-FS pairing, keeping the full flavour character natively — `[A₁,A₃]`
q² = the SU(2) adjoint `μ+1+μ⁻¹`; `[A₁,D₄]` q² = the U(2) currents
`1 + χ_{(1,−1)} + χ_{(1,1)} + χ_{(2,0)}`. The end-chord `[A₁,A₅]` and central-chord
E₇ agree through q⁵ and diverge at q⁶ (distinct theories); E₇'s UV BPS quiver is
certified E₇ (Cartan det 2). Self-test: `tests/test_flavoured_fork.py`.

This is the **ungauged sibling of the D-type ladder**: the gauged even-D rung is
`U1A1Deven` (in the parallel A1Dodd–U1A1Deven chain); `A1Deven` here is the same
`[A₁, D_{2k+2}]` *without* gauging the U(1) ⊂ U(2).

## The over-pure gauge-theory corner (Lagrangian SU(2) gauge theories)

The first **Lagrangian gauge theories** in this repository — built as RG flows over a
*pure-gauge* core (the AD theories above flow to gauged/ungauged AD survivors;
these flow to pure gauge theories, the matter integrated out):

    SU(2) + N_f fundamentals      →   pure SU(2)      (`SU2NfOverPure`)
    SU(2)×SU(2) + bifundamental   →   SU(2) × SU(2)   (`SU2xSU2BifundOverPure`)

Each matter block is the spectrum generator, peeled to pure-SU(2) Wilson
characters and integrated out, with the matter's flavour fugacities carried as an
`add_flavour(U(1)^…)` coefficient so the bilinear exact-FS trace is the
**μ-refined** Schur index. `SU2NfOverPure`: `S_RG = ∏_i E_𝖖(μ_i v)E_𝖖(μ_i/v)`,
flavour U(1)^{N_f} (the Cartan of SO(2N_f) — N_f=1 gives the U(1) current at q²,
N_f=2 the SO(4) adjoint `2 + Σ μ_1^{±1}μ_2^{±1}` = 6).

The bifundamental matter is the spectrum generator, integrated out:

    S_RG = ∏_{ε₁,ε₂ ∈ {±}} E_𝖖(μ · v₁^{ε₁} v₂^{ε₂}),

expanded over its four weights and peeled to SU(2)₁×SU(2)₂ characters
`χ_{w₁}(v₁)χ_{w₂}(v₂) → F⁽¹⁾_{w₁}F⁽²⁾_{w₂}` (Wilson lines of each pure SU(2)).
The auxiliary is `(pure SU(2) ⊗ pure SU(2)).add_flavour(U(1))` — two decoupled
pure-SU(2) cone K-algebras (from Step 2: `pure_su2_h_cone_data`, trace via
`pure_su2_h_trace_analytic`) with the bifundamental's **baryonic U(1)** as a
coefficient flavour μ. So the trace is the **μ-refined** Schur index in `R(U(1))`
(q² carries the baryonic `μ^{±2}` currents); pure exact-FS, no override.
The **SU(2)ⁿ linear quiver** (`SU2LinearQuiverOverPure(n, Nf1, Nfn)`) is the chain
composition — a bifundamental on each of the `n−1` links plus up to 2 end
fundamentals — over `(pure SU(2)^⊗ⁿ).add_flavour(U(1)^L)`, `L=(n−1)+Nf1+Nfn`; n=2
no-flavour *is* the bifund. Each link/end matter factor combines on its shared
node by SU(2) Clebsch–Gordan. (n≥3 traces are correct but slow — the generic
exact-FS pairing over the n-fold tensor.)

Self-test: `tests/test_over_pure.py` (`SU2NfOverPure`, `SU2xSU2BifundOverPure`,
and the `SU2LinearQuiverOverPure` n=2/n=3 construction).

## The SU(2)-gauged chain — nested auxiliaries (a flow as another flow's aux)

A fully-nested RG chain showing an `RGKAlgebra`'s auxiliary may itself be an
`RGKAlgebra`, composing all the way down to pure SU(2):

    SU(2)×U(1)-gauged A1D4 → SU(2)-gauged A1D3 → U(1)-gauged SU(2) N_f=1 → pure SU(2)

- `SU2Nf1PureSU2RGKAlgebra` — U(1)-gauged SU(2) N_f=1 over pure SU(2);
- `SU2GaugedA1D3RGKAlgebra` — aux = entry 1 (a flow used as the aux);
- `SU2U1GaugedA1D4RGKAlgebra` — aux = entry 2 ⊗ `QuantumTorusKAlg`, `S_RG =
  E_𝖖(X₀₁v)E_𝖖(X₀₁v⁻¹)` (gauge doublet on the fresh leg, χ-peeled to the pure-SU(2)
  Wilson reached *through* the nested aux).

All pure exact-FS over the (doubly-)nested aux. Entry 3's vacuum is
`1 − q² − q⁴ + q¹² + 2q¹⁴` (a q⁶–q¹⁰ gap), computed with this engine to q¹⁴;
the self-test pins the series to q⁸.
Self-test: `tests/test_su2_gauged_chain.py`.

## "Wild" formal flows — the framework decoupled from physics

The most general case: `S_RG = E_𝖖(L_a)` on **any monomial cone ray** `L_a` of
**any (product of) `ConeKAlgebra`(s)** gives a well-formed, truncation-safe
`RGKAlgebra` — closed `multiply`, orthonormal canonical basis, convergent `trace`
— **even when no 4d N=2 theory realises it**. The machinery is decoupled from
whether a "theory" exists; the only requirement on `L_a` is that it be a monomial
ray (`L_a^n` a single auxiliary label).

- `WildMonopoleRGKAlgebra` — `S_RG = E_𝖖(L_{1,0})` on the **'t Hooft monopole**
  ray of pure SU(2) (integrating out a *monopole* hyper — not a standard RG
  flow): `Tr(1) = 1 + 3q² + 9q⁴ + …`, clean and truncation-stable.
- `WildA1D3SquaredRGKAlgebra` — two `[A₁,D₃]` coupled by a fictional `E_𝖖(μ·L·L')`
  over `A1DoddConeKAlg(0) ⊗ A1DoddConeKAlg(0) ⊗ QT_μ` — a genuine **pair-of-cones**
  example.

Self-test: `tests/test_wild.py`.

## What's included (`src/rg/`)

Engine:
- `rgkalgebra.py` — the `RGKAlgebra` contract + the full derived `KAlgebra` API
  (the BPS/`rg_flow` JSON-serialization helpers raise if called — they require
  the BPS realisation layer, `src/bps/`, which this layer never imports);
- `grading.py` — the `Γ_RG`-grading sidecar (charge + height + cone);
- `graded_rg_solver.py` — the exact RG-discovery co-solver.

(Exact `(1−𝖖^{2n})`-localized-ring arithmetic `habiro.py`, palindromic 𝖖-number
polynomials `q_number_poly.py`, and lattice / pointed-cone utilities `lattice.py`
are shared with Step 2 and live in `src/cone/`; the RG layer imports them by
flat name.)

Reference flows:
- `u1_square_rg.py` — `U1SquareRGKAlgebra` (torus corner);
- `pentagon_square_rg.py` — `PentagonSquareSampleRGKAlgebra` (non-torus corner);
- `sqed_nf_rg.py` — `SQEDNfRGKAlgebra` (flavoured corner), with
- `sunf_dilog.py` — the `∏_i E_𝖖(μ_i x) → χ_{SU(N_f)}` spectrum tool.

A-type chain:
- `a1aeven_to_u1aodd_rgkalgebra.py` — `A1AevenToU1AoddRGKAlgebra(k)` (leg 1);
- `u1aodd_to_even_qt_rgkalgebra.py` — `U1A1AoddToEvenQTRGKAlgebra(k)` (leg 2).

D-type gauged chain:
- `a1d3_sqed2_rgkalgebra.py` — `A1D3Sqed2RGKAlgebra` (`[A₁,D₃] → SQED₂`);
- `u1a1deven_sqed_rgkalgebra.py` — `U1A1DevenSqedRGKAlgebra(k)`.

E-type flows:
- `e6_rgkalgebra.py` / `e8_rgkalgebra.py` — `E6RGKAlgebra` / `E8RGKAlgebra`;
- `u1a1e7_rgkalgebra.py` — `U1A1E7RGKAlgebra` (u(1)-gauged `[A₁,E₇]`).

Ungauged `add_flavour` fork:
- `a1aodd_to_even_rgkalgebra.py` — `A1AoddToEvenRGKAlgebra(k)` (U(1));
- `e7_rgkalgebra.py` — `E7RGKAlgebra` (ungauged `[A₁,E₇]`, U(1));
- `a1deven_rgkalgebra.py` — `A1DevenRGKAlgebra(k)` (U(2)).

Over-pure gauge theory (depends on Step 2's pure-SU(2) cone):
- `su2_nf_over_pure.py` — `SU2NfOverPure` (SU(2)+N_f);
- `su2su2_bifund_over_pure.py` — `SU2xSU2BifundOverPure` (SU(2)×SU(2)+bifund);
- `su2_linear_quiver_over_pure.py` — `SU2LinearQuiverOverPure(n, Nf1, Nfn)`
  (SU(2)ⁿ chain; n=2 = the bifund; n≥3 traces correct but slow).

SU(2)-gauged chain (nested-aux — each rung's aux is the previous rung's flow):
- `su2nf1_pure_su2_rgkalgebra.py` — `SU2Nf1PureSU2RGKAlgebra`;
- `su2gauged_a1d3_rgkalgebra.py` — `SU2GaugedA1D3RGKAlgebra` (aux = entry 1);
- `su2u1gauged_a1d4_rgkalgebra.py` — `SU2U1GaugedA1D4RGKAlgebra` (aux = entry 2 ⊗ QT).

Wild formal flows (depends on Step 2's cones):
- `wild_rgkalgebras.py` — `WildMonopoleRGKAlgebra`, `WildA1D3SquaredRGKAlgebra`.

Self-tests (in `tests/`): `test_rg_flows.py` (the three reference flows),
`test_a1an_chain.py` (the A-type chain), `test_dn_chain.py` (the D-type gauged
chain), `test_e_type.py` (E₆, E₈, gauged E₇), `test_flavoured_fork.py` (A1Aodd +
ungauged E₇ + A1Deven), `test_over_pure.py` (SU(2)+N_f, bifund, SU(2)ⁿ quiver),
`test_su2_gauged_chain.py` (the nested SU(2)-gauged chain), `test_wild.py` (the
wild formal flows).

## The spine-free guarantee

The whole `KAlgebra` API of every flow — structure constants, `ρ`, and the Schur
index (trace) to any `𝖖`-order — is computed with **no realisation engine** (no
`bps_kalgebra` / `rg_flow` / `chart_graph` / …). Seven of the eight self-tests
(all but `test_rg_flows.py`) assert that **no spine module is present in
`sys.modules`** after the run — a machine-checked spine-freeness certificate for
those suites.

## Tests

```bash
python3 run_tests.py        # the full gate (all four layers), from the repo root
```

`run_tests.py` puts every `src/<layer>/` directory on `sys.path` and runs the
Step-1/2 contract tests followed by the eight Step-3 RG self-tests and the
Step-4 BPS self-test. Each prints a `PASS` / `ALL … PASSED` line;
`test_cones.py` (Step 2) is the slowest (a few minutes).

## License

GPL-3.0-or-later (see `LICENSE`).
