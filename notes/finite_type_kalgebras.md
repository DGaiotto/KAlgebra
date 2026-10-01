# Finite-type KAlgebras — status, role, and the trace-uniqueness program

*(2026-06-10.  The coherent map of the finite-type corner: what exists,
why it matters scientifically, and the programs it anchors.  Code
pointers are to the canonical surface; plan pointers to
`restructuring_plans/`.)*

## 1. What "finite type" is here

The finite-type KAlgebras are the `A_𝖖[A₁[X_n]]` of the finite
Argyres–Douglas theories — `X = A_{2..7}`, `D_{3..8}`, `E_{6,7,8}` —
the corner of the catalogue with **finite BPS spectra, finitely many
cluster cones, and table-driven exact multiplication**.  Each lives in
up to four certified presentations:

| tier | where | what it is good at |
|---|---|---|
| frozen cone zoo | `finite_kalgebras/` registry + `implementations/finite_*_kalg.py` | uniform machine-generated standalones (cluster-BFS from the embedded BPS quiver); fast exact multiply; chord-geometry classifiers |
| hand-written closed forms | `PentagonKAlg`, `A1A2kKAlg(k)` (labels: multisets of pairwise non-crossing diagonals of the `(2k+3)`-gon, `curve(x, ell)`; ρ the rotation, cones the triangulations), `HexagonKAlg`, `A1DoddConeKAlg(k)` (and its presentation on the curves of the once-punctured `(2k+3)`-gon, `A1DnKAlg(2k+3)`), `A1D3/5KAlg`, `SU3ADKAlg` | independent derivations; closed-form Layer-2 traces (Rogers–Ramanujan, Andrews–Gordon, SU(2)/SU(3) characters) |
| BPS realisations | `BPSKAlgebra` on the embedded quiver literals | the universal exact oracle (Habiro Schur engine, chart graph, RG surface) |
| abstract object | `finite_kalgebras.objects.kalgebra_object(sid)` | the `KAlgebraObject` holding the above with certified `KAlgebraIso` witnesses |

Which class serves each ADE row (and the `U(1)`-gauged ones), and what a
fresh-process measurement finds on its serving path — stored data, runtime
oracles, trace windows, geometric labels — is the table of §9, guarded by
`tests/test_ade_serving_guard.py`.

## 2. Why this corner carries scientific weight

Everything the repo conjectures globally is *exhaustively checkable*
here, in exact arithmetic:

* **Goal 2.1 / 2.6 (orthonormality; the full Gram matrix).**  With the
  elementary-trace layer (below), `trace` / `inner_product` /
  `verify_orthonormality` run on the zoo through the universal
  contract surface; all-pairs Gram windows pass on six entries
  (`tests/test_finite_zoo_traces.py`).  The E-series Gram matrices are
  computable-by-runbook data nobody has looked at.
* **Goal 2.11 (trace miracle).**  The finite zoo is the systematic
  testbed; the frozen elementary traces double as the stabilisation
  targets.
* **Goal 1.3 (equivalences).**  One abstract algebra, several certified
  presentations — the `KAlgebraObject` epistemology was built and
  validated here first (§4).
* **Oracles for the frontier.**  Plans 22/24 lean on certified small
  nodes; the finite types are the cheapest such nodes.
* **Trace uniqueness (§5)** — the newest application: the contract
  *asserts* the ρ²-twisted trace is unique up to `1 + O(𝖖)` rescale;
  finite types turn that assertion into finite linear algebra.

## 3. Exact elementary traces = chiral-algebra characters

`finite_kalgebras/elem_traces.py` (+ `elem_trace_data.py`, empty since
2026-09-23: its last table, `e8`'s, gave way to the W₃(3,8) recipes and is
kept as a fixture of `tests/test_w3_seeds.py`; nothing falls back to the BPS
engine any more — see the status note at the top of `elem_traces.py`).
Layer 1 (the tagged-cycle reducer) reduces any trace on a cone
presentation to the **elementary seeds**: the identity and one
single-mult-gen seed per ρ²-orbit.  Those seed traces are the
**chiral-algebra characters** of the AD theory — M(2,5) Rogers–
Ramanujan for the pentagon, M(2,2k+3) Andrews–Gordon for the odd
polygons, flavoured characters for the D-series — generated *exactly*
from the embedded BPS quiver (Habiro/Nahm engine), frozen per entry
(pentagon K=64 … e6 K=12), validated against every independent closed
form in the repo, and extended lazily-exactly past the frozen window.
*(2026-09-23: on the user's ruling that frozen data a working route
reproduces is pointless — Plan 42 `decisions.md`, R14 — every table but
`e8`'s was removed after its live route reproduced it exactly; the zoo now
serves closed forms, or the Nahm-sum vacuum plus the orthonormality
bootstrap.  Later the same day `e8`'s went too (R19), and the per-seed BPS
fallbacks became `NotImplementedError`s naming the entry, the seed and the
order; the su2u1 entries `a1d6` / `a1d8` serve no trace and name the class
that computes it — `a1d4` is served from `SU3ADKAlg`, see "Trace routing"
below.  Since 2026-09-24 `a1d6` / `a1d8` are served too, from `A1DevenKAlg(2)` /
`A1DevenKAlg(3)`.)*

**BPS-free generation by the orthonormality bootstrap (2026-06-14,
`experiments/e6_elem_bootstrap.py`).**  The BPS Habiro/Nahm engine is
heavy on the E-series (~200 s per e6 seed at K=8, ~1200 s for the six
seeds, hours at K≳16).  The SU3AD deconvolution transplants to the
finite corner and removes it: given **only Tr(1)**, the six seeds are
reconstructed by exact linear algebra over the *cheap* cone-data
Layer-1 reductions, no BPS.  Trivial-R ⇒ plain integer q-series.  Each
canonical L≠1 has Tr(L)=O(q); for the deep single-mult-gen labels
`((i,a),)` the reducer gives `Tr = Σ_s P_{i,a,s}(𝖖)·Tr(s)` (6 seeds +
identity) with a mult-gen-dependent negative reach, and every q^{≤0}
coefficient that stays closed on Tr_j[1..K] is one exact equation.
The identity-pairings pin **5 of 6** seeds; the one non-leading seed
(mg0) is completed by the general orthonormality pairs
`I_{La,Lb}=δ+O(q)` over deep mg0-labels (degree-≤6 products stay inside
the reducer).  The combined system is hugely over-determined and
CONSISTENT (501 eqns / 72 unknowns at K=12 — a BPS-free certificate),
**matches the frozen e6 table exactly** in ~4 s, and extends past the
frozen window (confirmed at K=14 with Tr(1) from one BPS call).  Reach
ceiling: the cone-data reducer caps near degree 6 (`((0,7),)` does not
terminate), bounding the top order.

**u1 generalisation: the BPS-free trace for a5/a7/e7 (2026-06-14,
`finite_kalgebras/u1_bootstrap.py`, the u1 path of `elem_traces.generate`).**
The bootstrap extends to the u1 (abelian-flavour) entries via the SU3AD
μ-fugacity treatment: seed traces are μ-Laurents, the reduction coefficients
`P_{L,s}(𝖖,μ)` carry μ-charge, and `Tr(L)=O(q)` is imposed q^{≤0}- AND
μ-charge-wise.  Two ideas make it scale: (i) **ρ²-orbit reduction** — within a
ρ²-orbit the seed traces are μ-power multiples of one rep
(`Tr(L_{ρ²i})=μ^{δ(i)−δ(ρi)}·Tr(L_i)`), collapsing the unknowns to one rep per
orbit (e7: 90 seeds → 18 reps); (ii) a **forward-triangular** per-order frontier
sweep (run to K+margin, truncate — the top orders are boundary-under-constrained).
The deep identity-pairings pin most reps; the few free reps get the δ-honest
orthonormality pairs `I_{La,Lb}=μ^{δ(a)}·(label-pair)`.

**e7 — recorded SOLVED on 2026-06-14; NOT reproduced.**  *(Correction,
2026-09-22 — repo_audit.md C.errata item 8(b): e7 has no frozen table; the
claim below that the bootstrap pins all 90 seeds was not reproducible in 31
minutes of CPU; and a single e7 seed at K=4 DOES finish by the per-seed BPS
engine, in 203 s cold.  The text below is the 2026-06-14 record.)*
*(2026-09-23: reproduced by the corrected seed bootstrap, Plan 42 R17 route 2 —
repo_audit.md C.errata item 16, "`u1_bootstrap._sweep` reads never-solved seed
values as 0" — which pins all 90 seeds through 𝖖¹⁴; the zoo serves them through
it, `Tr(1)` and every seed at `K = 8` in about 70 s.  Since the same day the zoo
serves e7 from closed forms instead — every seed a theta-product combination and
`Tr(1)` the Bershadsky–Polyakov vacuum product, `finite_kalgebras/e7_seeds.py`
(R17 route 3; Plan 42 `finite_type_trace_machinery.md` section M) — with the
bootstrap as the witness.)*
e7's standalone shipped without a `_rho_delta` table (so its
u1 reductions were not δ-honest and the bootstrap went inconsistent);
`experiments/e7_compute_rho_delta.py` recomputes it the way a3/a5 got theirs
(δ(i)=flav(ρ(γ_i))−flav(canon ρ_perm(i)) over the BPS flavour basis, validated
by reproducing the frozen a3/a5 tables) and it is now written into
`finite_e7_kalg.py` (`E7_RHO_DELTA`, 41/90 nonzero).  With it the bootstrap pins
all 90 e7 seeds (over-determined + consistent); `FiniteE7KAlgebra.trace` now
uses the δ-honest Layer-1 path → the bootstrap-served seeds (the per-seed BPS
engine is infeasible: a single e7 seed trace doesn't finish at K=4).  Validated
BPS-free vs frozen: a3 6/6, a5 24/24, a7; e7 `Tr(1)`/`Tr(seed)` exact.

**e8 — SOLVED (2026-06-14, trivial-R, the incremental degree-sweep;
`experiments/e8_elem_bootstrap.py`).**  e8 is the largest exceptional (16
ρ²-orbit seeds, trivial R).  The e6 full-Gaussian path (`_generate_bootstrap`)
*timed out* (>900 s) on it: for each non-leading seed it blindly generated
EVERY orthonormality pair (powers a,b≤3, all cross-seeds), and the deep
mixed-monomial reductions grind to the Layer-1 200 k step cap — the **e8
"degree-4 wall"** (single-power and low-degree pair reductions are cheap;
degree-4 *mixed* e8 cone monomials do not terminate).  The fix has two parts.
(i) Generate the pairs **cheapest-total-degree first, re-solve after each
degree, and stop the moment no seed is free** — a seed is pinned at q-order k by
a pair reaching emin≤−k, so the depth needed grows with k and the sweep adds
only what is necessary.  (ii) Range the pair's **first factor over the FULL seed
set**, not just the currently-free seeds: a free seed's trace can be pinned by a
pair whose two factors are OTHER seeds (the Layer-1 reduction of L_idx^a·L_jj^b
produces the free seed even when neither factor is it), so the original
free-first restriction under-constrains the deeper windows.  With both, **e8
closes at degree 3 through K=6 — entirely wall-free** (~13 s of pairs over the
one BPS `Tr(1)` call at Ki=8 ≈ 50 s; the degree-4 wall is never touched).  All
16 seeds pinned, over-determined + consistent (the BPS-free certificate); e8
`Tr(mg27)` independently matches the genuine BPS-quiver trace (`−q`), the K=6
window agrees with the K=4 window on its overlap, and e6 still matches its
frozen table 6/6 under the same code.  Served like e7 (the per-seed BPS engine
is infeasible) via the new **trivial-R on-demand path** in `_seed_series`
(`_TRIVIAL_REC`, mirroring the u1 `_U1_REC`), and **frozen at K=6**;
`FiniteE8KAlgebra.trace` already routes through the Layer-1 → bootstrap path.
Past K≈6 both the wall (deg≥4 reductions) and the steeply-growing BPS `Tr(1)`
cost gate further depth.  (The full-window `verify_orthonormality` is likewise
compute-gated for e8/e6 — the contract's trace-order widening pulls in a high-K
BPS `Tr(1)` and the deg-4 wall — so, as for e6, the certificate is the
over-determined consistency plus the BPS spot-check, not the Gram window.)

**Trace routing (which source per u1 entry).**  `elem_traces.generate`/
`_seed_series`: **a3/hexagon use the exact closed-form characters**
(`ad_characters.a3_elem_entry`); **a5/a7/e7 use the bootstrap** (no exact
characters).  All 44 `test_finite_zoo_traces` tests pass.  *(2026-09-23: e7 now
from its closed forms, `finite_kalgebras/e7_seeds.py`; later the same day
a3 / a5 / a7 (and hexagon / octagon / decagon) from the ungauged polygon
`ungauge_u1a1aodd(k)` (`finite_kalgebras/aodd_seeds.py`), a1d3 from
`A1DoddConeKAlg(0)` (`a1d3_seeds.py`) and a1d4 from `SU3ADKAlg` restricted to
SU(2)×U(1), with the zoo's U(1) fugacity the cube of the branching one
(`a1d4_seeds.py`) — generator maps built at runtime and certified by every
generator product; the u(1) and su(2) bootstraps and the a3 characters are
witnesses now.  Record: Plan 42 `finite_type_trace_machinery.md` section L.)*
*(2026-09-24: a1d6 / a1d8 from `A1DevenKAlg(2)` / `A1DevenKAlg(3)`
(`finite_kalgebras/a1deven_seeds.py`) — the zoo's U(1) fugacity is the
ungauged gauge fugacity itself; the map is certified by every generator product
and by ρ, and the traces reach as deep as `A1DevenKAlg`'s trace transport does
(every generator through 𝖖¹² / 𝖖⁸) and raise past it.  Record: section L.  Later
the same day the seeds' closed forms (`u1a1deven_seed_characters`, measured)
took the generator traces off the transport: every seed and `Tr(1)` to any
order, all of them through 𝖖⁴⁰ in 1.1 s / 2.9 s after the generator map.
Record: section J.)*

**The Verlinde q→1 conjecture is CONFIRMED exactly on four theories**
(2026-06-10; `experiments/verlinde_q1_probe.py` +
`tests/test_verlinde_q1.py`): writing each trace in the character
basis via the certified dictionaries, the line-defect classes close
on the chiral Verlinde fusion ring — `[L_a L_b] = [L_a] × [L_b]`
holds for every degree-≤1 window product — pentagon 36/36, heptagon
225/225, nonagon 784/784 on M(2,2n+3), and **a3/[A₁,A₃] 49/49 on the
su(2)_{−4/3} ring including its negative-level signs**
(`[Φ₂]² = −[Φ₁]`, `[Φ₁][Φ₂] = −[Φ₀]`) — with structure constants and
Layer-1 reductions evaluated at q = 1 in exact integer arithmetic;
plus the **cross-implementation check**: a1d3 = [A₁,D₃] ≅ [A₁,A₃]
(different cone data, su2-refined, folded seeds) closes 49/49 on the
same ring (su2 characters at μ=1 evaluated as DIMENSIONS via
`to_abelian`).
Two structural notes: (i) the naive numeric route — q→1⁻ trace
ratios — is a 0/0 of exponentially small alternating thetas; the
class formulation needs no asymptotics at all; (ii) on flavoured
entries q = 1 is also μ = 1, where the Plan-18 μ^δ slide leak is
exactly 1 — the Verlinde specialization is leak-immune, which is why
the u1 entry a3 is checkable today, ahead of the Z-form regeneration.

**Closed forms now live in `finite_kalgebras/ad_characters.py`**
(2026-06-10, from the PR #424 defect-index harvest): the three
su(2)_{−4/3} admissible characters assembled into [A₁,A₃]'s line
traces `I = χ₀`, `I_A = q^{−1/2}z⁻¹(−χ₁+χ₂)`,
`I_B = q^{−1/2}(χ₀−χ₁+z⁻²χ₂)` (dictionary `q_paper = 𝖖²`, `z² = μ`,
per-seed torsor `z^c`), and all M(2,2n+3) characters in
vortex-residue theta form.  The **a3 frozen entry is generated from
these closed forms** (K=48; `experiments/a3_refreeze_from_characters
.py`), and the pentagon/heptagon BPS-frozen tables are re-certified
against M(2,5)/M(2,7) over their FULL windows
(`tests/test_ad_characters.py`).

Three flavour theorems-by-experiment came out of wiring this
(recorded in Plan 18 notes; all verified on a3):

1. **ρ²-orbit folding of seeds is invalid for unit-character flavour**
   (orbit traces differ by `μ`-shifts); valid for trivial/su2.
2. **The Layer-1 cyclic slides drop `μ^δ` on unit-character entries**
   (label-level ρ² ≠ element-level ρ²) — so u1/su2u1 standalones
   bypass Layer 1, tracing every label directly at its γ-charge.  The
   Plan-18 Z-form regeneration dissolves the leak by construction.

3. **The flavoured BPS trace at q-window K is exact only on a
   trapezoid** in (𝖖-order k, μ-charge): flavour tails at orders near
   the top of the window need internal q-orders beyond K and come out
   clipped, or with spurious uncancelled edge terms (empirical wedge
   `|μ| ≲ (2/3)(K−k)`, route-dependent).  Caught 2026-06-10 by the
   closed-form cross-validation: the retired BPS-frozen a3 table
   violated the cyclicity identities `T₀ = T₃`, `T₄ = μT₀` from 𝖖²³
   on; the live oracle at K=24 corrupts its 𝖖²⁴ row beyond |μ| ≥ 3,
   while at K=44/48 it reproduces the characters exactly
   (`experiments/a3_character_check.py`).  Consequences: flavoured
   frozen tables are trustworthy only strictly inside their wedge —
   a5 (K=24), a1d3 (K=40), a1d5 (K=32) carry the same latent tail
   defect until regenerated with margin or from closed forms; the
   lazy extension path inherits the defect near the top of every
   requested window; the root cause lives in the windowed
   `rg_generator` route of `trace` and is logged for an upstream fix.

Remaining: E-series deep tables (offline jobs; runbook in the module),
su2u1 entries (gated on Plan-18 PR2), the a1d4 rebuild
(`a1d4_kalgebra()` via `su3_to_su2u1_hom` — functorially certified,
inheriting the **known upstream SU3AD defects**: ρ-automorphism
failures + vacuum-character inconsistency, asserted in tests).

## 4. The object layer (one object, many presentations; one flow, many descriptions)

* `kalgebra_object.py::KAlgebraObject` — realizations + verified
  `KAlgebraIso` witnesses as a connected groupoid: composed transport,
  capability routing, per-witness batteries, **path-independence
  coherence** (isos form an `Aut`-torsor — coherence certifies the
  curation, not a law of nature).
* `bps_chart_object.py` — chart mutations as witnesses (same algebra,
  *different flow*); the **monodromy result**: the pentagon's pure
  rotation closes the chart data in 4 necklace steps with composed
  label map **exactly ρ²** — the twisted-trace ρ² realized as the
  chart-graph loop, with the naive loop-closure correctly rejected as
  incoherent.
* `rgkalgebra_object.py::RGKAlgebraObject` — the flow-typed
  refinement (witness = UV iso + aux iso + RG/S_RG/apex/grading
  verifiers): frame changes are the same flow (battery passes,
  gradings intertwine via `Λ = T`), chart mutations are not
  (`verify_s_rg_match` fails), and the `SubquiverRG`-vs-
  `DirectionalSubquiverRG` pair turns the F-oracle/co-solver
  cross-check into a curated certificate.
* `restructuring_plans/26_rg_mutation/` — the research program to
  extend mutation to arbitrary RGKAlgebras
  (`S_RG = S₁S₂ ⟼ S₂·ρ_IR⁻¹(S₁)`): formal core derived and
  convention-pinned; C1–C5 open; BPS instance tested.

## 5. The trace-uniqueness program (begun 2026-06-10)

**The question.**  The contract treats `Tr` as data subject to
ρ²-twisted cyclicity and `q⁰`-orthonormality, and asserts it is
"defined up to an overall rescaling by `1 + O(𝖖)`".  Is that the whole
truth?  On a finite-type algebra this is a *finite, exact* question:
on any label window the admissible traces form the solution set of a
linear system over `Q`, order by order in `q`.

**The harness** (`trace_uniqueness.py`):

* unknowns `t_c[k]` (trace coefficients; `Z[[q]]` support assumed —
  true of every known finite-type trace);
* homogeneous rows: pair cyclicity `Tr(ab) − Tr(ρ²(b)a) = 0` for
  window pairs, expanded per order with **no silently-truncated
  equations** (unknowns are carried to `K_unknown ≈ 2K_eq + 4`);
* inhomogeneous rows: `q⁰`-orthonormality on window pairs;
* **trace-bootstrap rows** (user, 2026-06-10): 4-point crossing
  `Tr((L_aL_b)(L_cL_d)) = Tr((ρ²(L_d)L_a)(L_bL_c))` — channel
  `(ab|cd)` vs `(ρ²(d)a|bc)` — which imports the deep labels'
  cyclicity relations while keeping all four external legs in the
  window;
* exact Fraction Gaussian elimination; the *projected* statement is
  measured (kernel restricted to (window labels, orders ≤ K_eq)):
  closure-only labels are nuisance unknowns, so the full kernel is
  window-inflated by construction; the window's honest content is the
  solution set's shadow on its own coordinates.

**The sharp conjecture under test** (`verify_unique_up_to_rescale`):

    projected admissible traces  =  projected { f·Tr : f ∈ 1 + q·Q[[q]] }

i.e. (i) the canonical trace solves everything, (ii) projected kernel
dimension = K_eq, (iii) the kernel is exactly the rescale line.

**First results (pentagon).**

* `K_eq = 2`, degree-≤2 window (16 labels): projected dim
  **= 2 = K_eq**, kernel = rescale line — uniqueness up to rescale
  *verified*.
* `K_eq = 4`, degree-≤2 window: projected dim = **5** — one
  ρ-invariant mode beyond the rescale line (`+q³` on all five
  `Tr(L_i²)`, `+q⁴` on all five adjacent `Tr(L_iL_{i+1})`), invisible
  to every window pair's `q⁰`-orthonormality, and **surviving the
  4-point bootstrap with generator legs** (625 quadruples; they do
  cut the full kernel 50 → 40 but miss the killing relations).
* `K_eq = 4`, degree-≤3 window at full unknown depth
  (`K_unknown = 12`, 1378 unknowns, 7021 equations): projected dim
  **= 4 = K_eq**, kernel **= the rescale line** — *uniqueness up to
  `1 + O(q)` rescale verified through `q⁴`*.  The q³ mode was window
  under-constraint, not mathematics.

**Methodological lessons** (pinned in
`tests/test_trace_uniqueness.py`): (a) windowed kernels can mimic
genuine deformations — discriminate by *both* deepening the unknown
orders (`K_unknown ≫ K_eq`; truncation drops equations silently
otherwise) *and* widening the pair window; (b) the v1 bootstrap
channels (homogeneous 4-point cyclicity crossings) are measurably
weaker than window-widening on this mode: generator legs miss it AND
square-leg quadruples `(L_i², g, g, g)` miss it (243 s, dim still 5),
while the degree-3 *windowed system* kills it — plausibly through its
`q⁰`-orthonormality pairs with degree-3 legs, which the bootstrap (as
homogeneous crossings) never adds.  Open refinement: bootstrap the
*inhomogeneous* data too (channel-factorized orthonormality), and
locate exactly which deg-3 rows are load-bearing.

**Heptagon:** K_eq = 2 uniqueness-up-to-rescale **verified** on the
3-cone + squares window (25 labels, ~25 min exact Gaussian; pinned
slow-gated); small windows under-constrain by one mode, pinned as the
fast consistency test.

**v2 — the Gram bootstrap (user direction, 2026-06-10).**  Until now
traces were constrained recursion-style: Layer-1 cyclicity iterates
trace data at `q^k` to `q^{k+1}`, and the v1 system above keeps the
1-point values `t_c` primary — where the v1 4-point rows measurably
underperform window-widening.  The bootstrap framing of other
bootstrap problems makes the **2-point data primary**:
`G_{ab} = Tr(L_a L_b)` as the unknown layer, with

    2pt cyclicity      G_{ab} = G_{ρ²(b),a}
    orthonormality     G_{ρ(a),b}[q⁰] = δ_{ab}     (DIRECT boundary data)
    4pt crossing       Σ C^e_{ab}C^f_{cd} G_{ef} = Σ C^e_{ρ²(d)a}C^f_{bc} G_{ef}

(`gram_bootstrap_space` / `verify_gram_unique_up_to_rescale`).  Two
structural advantages: orthonormality constrains the unknowns at q⁰
*directly* (in 1-point coordinates it enters only through deep
C-mixing), and the system needs **only structure constants on a
window** — no Layer-1, no cone presentation, no 1-point closure — so
the same equations run on infinite-type algebras (pure U(N)) where
recursion-style trace bootstraps do not close.

**v2/v3 measured ledger (pentagon, q³-mode discriminator; pinned in
`tests/test_trace_uniqueness.py::test_gram_bootstrap_ledger`):**

| axiom set | deg-2 boundary | deg-3 boundary |
|---|---|---|
| v1 pairs (cyclicity + ortho) | dim 5 | **rescale line** ✓ |
| crossing only (legs: gens / +squares / +cubes; up to 14 641 quads, 151 736 eqs) | dim 5 | dim 5 |
| **v3 = crossing + OPE** | dim 5 | **rescale line** ✓ (4.7 s) |

Conclusions: (i) **crossing alone never pins the trace** at any tried
leg degree — the OPE half (`ope_reduction=True`: factorization through
the identity channel, `G_{ef} = Σ_c C^c_{ef}·G_{c,id}`) is independent
load-bearing data, exactly as in other bootstrap problems (crossing +
OPE data, not crossing alone); (ii) with OPE the Gram-primary system
reproduces the v1 verdicts in the window-free language; (iii) the
**reach of the q⁰ boundary data is the resolution knob** — degree-2
boundary leaves the q³ mode in every axiom set, degree-3 kills it.

**Engineering note:** the harness's exact elimination is sparse
(dict-rows + live column index); the dense→sparse swap took the
pentagon deg-3 verification 272 s → 1.0 s and the heptagon
verification 1533 s → 0.9 s, un-gating all verifications in the
default suite (13 checks, < 1 min).

**The fixed-𝖖 numerical bootstrap and the ρ wrinkle (user, 2026-06-10;
`experiments/pentagon_numeric_bootstrap.py`).**  At fixed real 𝖖 with
|𝖖| < 1 there is no "q⁰ part": the series-side normalization is
replaced by the **Schur quantization conjecture** (Goal 2.2) — the
pairing `(a|b) = I_{ab}(𝖖)` is positive definite — and the problem
becomes a conventional numerical bootstrap (linear cyclicity +
semidefinite Gram ⇒ feasibility islands).  The pentagon is the
cleanest lab: Layer 1 makes every cyclic functional `λ·Φ₁ + μ·Φ_L`,
and PSD is scale-invariant, so the whole bootstrap lives in the single
ratio `r = Tr(L)/Tr(1)` per 𝖖, with island `[r₋(𝖖), r₊(𝖖)]` around
the exact Rogers–Ramanujan ratio.

Two structural deviations from the conventional bootstrap, both
load-bearing:

* **Crossing is ρ²-twisted** (`Tr(abcd) = Tr(ρ²(d)·abc)`): the
  crossing orbit of one 4-point function has `4·ord(ρ²)` channel
  presentations (20 for the pentagon), not 4 — the twist *enriches*
  crossing.  Channel leg-sets must be **ρ-closed** for the full
  twisted orbit's equations to be represented (the generator sets used
  here are; this is a requirement to check, not an accident).  The
  chart-rotation monodromy result (§4: the full rotation composes to
  ρ²) is the same twist in wall-crossing language — twisted cyclicity
  IS monodromy invariance of the trace.
* **Reflection positivity is ρ-twisted**: `(a|b) = Tr(ρ(a)b)` is
  symmetric at real 𝖖 iff `Tr∘ρ = Tr` — true for the pentagon
  (verified exactly at the Fraction level).  On the **unit-character
  D-series** it is instead **Hermitian**: a3's ρ-orbit traces differ by
  `μ ↦ μ⁻¹` (§3), which is exactly `Tr∘ρ = ⋆∘Tr` — axiom 5 of the contract
  (`kalgebra.md` "Axiom 5", user 2026-09-18; first read this way in
  `repo_audit.md` A73), not a broken symmetry.  The SDP must then be posed
  on the Hermitian form (unitary `μ`).  e6 is unflavoured and sits on the
  symmetric side: `Tr∘ρ = Tr` holds there (A73, 568/568; the ρ-paired seeds
  `T₂ = T₆`, `T₅ = T₇` are exact in the frozen table).  The former "FALSE on
  e6" reading compared seeds 4 and 5, which lie in *different* ρ-orbits —
  corrected 2026-09-18, `repo_audit.md` A73.
  Measured on contract-compliant presentations of the same theories
  (#1509, landed in parallel: BPS flavoured hexagon 27/27 with `⋆` needed on
  20; `add_flavour(U(1))` 32/32; SQED₃ over a rank-2 flavour ring 54/54) —
  so the form is Hermitian as it stands and needs no symmetrization; the
  a3/a5 cone standalones fail the label form only because their `ρ` drops
  the flavour bookkeeping (`repo_audit.md` A74).  Verifiers:
  `KAlgebra.verify_trace_intertwines_rho_star` (alias
  `verify_trace_intertwines_rho`).

Numerics discipline note: the Gram entries are O(1) only after exact
cancellation of `q^{−m}`-sized Layer-1 contributions (m up to ~8 on
the deg-2 window, ~15 at deg-3) — float evaluation is catastrophically
wrong at small 𝖖 (a first float run produced a spurious "positivity
violation" of size 10²); entries must be computed in exact rational
arithmetic at rational 𝖖 and only the final balanced Cholesky run in
floats.  The repo's exact-arithmetic rule applies *pointwise*, not
just to series.

**Fixed-𝖖 results (pentagon; `experiments/pentagon_numeric_bootstrap.py`
+ `pentagon_fixed_q_bootstrap.py`, data in `experiments/data/`).**

* *Exact-solution + unitarity sweep* (18 points, both signs of 𝖖,
  exact rational Gram entries, balanced-Cholesky PSD, dyadic
  bisection): Schur positivity holds pointwise at the exact trace
  everywhere sampled; the deg-2 island width is **non-monotonic** —
  ~10⁻¹¹ at 𝖖 = 0.2, peaking ≈ 6.5·10⁻⁵ near 𝖖 ≈ 0.7, then
  *re-tightening* to 2.8·10⁻⁷ at 𝖖 = 23/25 as the Gram's approach to
  quantization degeneracy itself pins deformations.  Rigidity at BOTH
  ends (series lever-arms at weak coupling, near-null vectors at
  strong); the open-island "conventional bootstrap regime" is
  intermediate coupling.  Negative 𝖖 mirrors positive exactly; the
  Gram stays better-conditioned at 𝖖 < 0.
* *Genuine crossing bootstrap* (unknowns = `G_{ef}(𝖖)`, Layer 1 used
  only for the reference point): at fixed 𝖖 on the deg-2 window with
  id+generator channels, 2-pt cyclicity alone leaves a dim-28 affine
  space with PSD-unbounded flat directions; **adding the 4-pt crossing
  equations collapses it to dim 2 and bounds the island in every
  direction** (extents 0.86 / 0.38 / 0.019 at 𝖖 = 0.3 / 0.5 / 0.7 —
  shrinking with coupling).  With the OPE axiom on, crossing adds
  nothing (identical dim-2 islands): **pointwise, crossing ≡ OPE
  modulo 2-pt cyclicity on this window**, whereas the power-series
  side separates them sharply (the q³-mode discriminator) — the
  series-side OPE advantage lives entirely in the q-order mixing.

**Machine-generated uniqueness proofs, one algebra at a time
(`trace_uniqueness_proofs.py` + `tests/test_trace_uniqueness_proofs.py`;
user direction: with cyclic reduction to finitely many seeds, the
simple method wins).**  Per algebra the prover emits a full
certificate: (1) the seed-reduction differences of `Tr(ab)` vs
`Tr(ρ²(b)a)` — identically zero (free seeds) or consumed as certified
cyclicity rows; (2) the exact-rank certificate that
{cyclicity + q⁰-orthonormality over an auto-grown pair window} has
solution space of dimension EXACTLY N — the `1+q·Q[[q]]` rescale line
and nothing else; (3) the canonical chiral-character trace solves
everything and the kernel is its shift span.

| algebra | seeds | proven through | window | cyclicity relations |
|---|--:|--:|---|--:|
| pentagon | 2 | q⁸ | deg 3 (961 pairs) | 0 (free seeds) |
| heptagon | 3 | q⁶ | deg 2 (7 225 pairs) | 0 (free seeds) |
| e6 | 7 | q⁶ | deg 2 (234 256 pairs) | 221 906 rows |
| e8 | 17 (14 avail) | q⁴, projected onto available-seed cells | deg-1 (16 641 pairs) + 6 biting deg-2 pairs | 16 688 rows |
| nonagon* | 4 | q⁶ | deg 2 (78 400 pairs) | 0 (free seeds) |
| a1d3 (su2) | 3 | 𝖖⁴ × typed tower | deg ≤ 4, stable | 0 (free seeds) |
| a1d5 (su2) | 3 | 𝖖⁴ × typed tower | deg 2 | 0 (free seeds) |
| a3 (u1) | 7 | 𝖖⁴ × u1 tower | deg 3 (δ-honest) | consumed as rows |
| a5 (u1) | 25 | 𝖖⁴ × u1 tower | deg 2 (δ-honest) | consumed as rows |

**u1 certificates (2026-06-11, the δ-honest arc).**  The Plan-18 μ^δ
leak is CLOSED at the Layer-1 reducer: `cone_data._rho2_twist_unit`
threads the central units the tagged-cyclicity slides used to drop
(`ρ²(L_t) = μ^{δ(Pt)−δ(t)}·L_{P²t}`, cross-checked against the
#428–#430 z-form's operational ρ²; default-inert — only classes
binding `_rho_delta` are affected, trivial/su2 byte-identical).
Validated: the δ-honest reduction reproduces the zoo traces exactly
on the trusted wedge, and where they differ (deg-2 squares at the top
window order) the ZOO's bypass is the wrong one — its own A9 wedge,
with the reduction assembling exactly from the certified K=48
character tables.  `prove_uniqueness_u1_reduced` (honest reduction
for the seed collapse + Z-form ρ for the rows) then certifies a3
(deg-3, 40 s) and a5 (deg-2); u1 typing is the HALF-SLOPE cone
`2|m| ≤ k + c` with all integer charges and no parity pair (that was
su2 weight-lattice physics), and the u1 tower keeps ±j shifts.
a7's deg-1 system is consistent (canonical solves); its deg-2 window
is combinatorially heavy (65 mult-gens) — certificate pending.

**e8 (2026-06-11, user-ruled partial certificate)**: stage 1c is
A10-bounded — three deep-charge seeds (witness depth −11…−15) sit
behind measured memory walls (F-solves > 8 GB for two; a > 13 GB
Schur walk for the third) even after the A10 η-shell fix made the
other 14 routine (the heaviest banked seed: 10 min / 2.4 GB, vs the
pre-fix 16 GB OOM).  The certificate
(`experiments/data/e8_proof_certificate.json`, runbook
`experiments/e8_proof_run.py`) imposes EVERY axiom row (31 271; the
missing seeds' cells enter as unconstrained unknowns), checks the
canonical against the 21 619 rows fully supported on known cells, and
proves the projected statement: on the available-seed cells the
solution space is exactly the rescale line (dim 4 = N).  Like e6,
e8's cyclicity forces seed relations (16 688 nonzero rows).
Completing the three seeds = the A10 engine program (windowed
F-solve / banded s-table), a standing item.

(*) the nonagon = A₆ AD = `A1A2kKAlg(3)` — not a frozen-zoo entry;
its instant cone traces are the four M(2,9) characters, delivered in
closed form by the vortex-residue formula and pinned in
`tests/test_ad_characters.py` (`Tr 1 = χ₍₁,₁₎`, seeds
`q⁻³(χ₍₁,₃₎−χ₍₁,₄₎)`, `q⁻¹(χ₍₁,₁₎−χ₍₁,₂₎)`, `q⁻²(χ₍₁,₃₎−χ₍₁,₂₎)`
through 𝖖³⁰).  a1d7 (su2, unfrozen) is generation-gated: its lazy
seed traces are E-series-scale (≈4 GB / >12 min at K=6, killed in
favour of the e8 run); freeze first, then the flavoured prover
applies as-is.

**The golden-standard family, both halves, parametrically
(`experiments/a1a2k_trace_bootstrap.py` + `tests/test_a1a2k_trace_bootstrap.py`,
2026-06-22).**  For `A1A2kKAlg(k)` = `A_𝖖([A_1, A_{2k}])` the two halves of
Goal 2.12 now run together, on the *standalone cone class itself* (spine-free)
and parametrically in `k` (pentagon/heptagon/nonagon = k=1,2,3): the
**producer** (`trace_bootstrap.verify_bootstrap`, the §3 order-gain isolation)
derives every elementary trace in the deg-≤2 cone window from the vacuum seed
`Tr(1)` alone — 45 / 693 / 5667 traces, **0 stalls, 0 conflicts** — and each
derived `Tr(L_c)` equals the closed-form M(2,2k+3) Andrews–Gordon character
`A1A2kKAlg(k).trace`; the **verifier** (`prove_uniqueness`) certifies the
admissible-trace solution space is exactly the `1+𝖖·Q[[𝖖]]` rescale line with
**0 cyclicity relations** (free seeds — the property that makes this family the
paragon, vs e6/e8's forced seed relations).  So on the gold standard the
explicit trace is constructively reproduced *and* certified the unique solution
of cyclicity + orthonormality, up to the documented `Tr(1)` rescale (whose
pinning is the separate 2.11 miracle).

**Toward all orders (`a1a2k_all_orders.md`, `experiments/a1a2k_all_orders.py`,
2026-06-22).**  The per-N certificate is already the all-orders-*truncated*
theorem; the **proven** rescale lemma (`f·Tr_can` is admissible for any
`f∈1+𝖖·Q[[𝖖]]`) + the kernel-is-rescale-line check elevate it to:
**all-orders uniqueness ⟺ the certifying window degree `d(N)` is finite for
every `N`**.  There is no fixed-window certificate (a deg-3 pentagon window
certifies only `N≤14`); the window grows like `√N` (deg-2→`N≤4`, deg-3→`N≤14`,
deg-4→`N≥34`) — finite at every order.  Full certificates (kernel = rescale
line) are pushed to **q³² (pentagon)** / **q¹² (heptagon)**, past the q⁸/q⁶
pins above.  The residual open core is exactly `d(N)<∞ ∀N` — the Goal-2.11
stabilisation phenomenon.

**Flavoured proofs (`trace_uniqueness_flavoured.py` +
`tests/test_trace_uniqueness_flavoured.py`).**  For su2-flavoured
entries the unknowns become μ-graded seed cells `T[(seed, k, m)]`
(su2-character coefficients abelianized; honest label-cyclicity —
no unit-character torsor), and the headline is a **new structural
result**: cyclicity + q⁰-orthonormality alone do NOT pin the trace.
The windowed systems plateau (pair degree ≤ 6, grids 𝖖⁶–𝖖¹⁰) with
stable extra functionals — every one supported on cells violating the
**support typing** of the canonical trace:

* **Weyl**: coefficients are su2 characters (μ ↔ μ⁻¹ symmetric);
* **parity pair**: two independent ℤ₂ charge-conservation rules —
  fermion parity fixes `k mod 2` per seed, flavour-charge parity
  fixes `m mod 2` per seed (a1d3: identity (0,0), T₀ (1,0),
  T₁ (1,1));
* **spin-charge cone**: `|m| ≤ k + c(seed)` — a unit of flavour
  charge costs at least a unit of 𝖖 in the Schur measure.

All three are charge-conservation theorems of the γ-graded BPS
realisation.  **Correction (user ruling 2026-07-24): this is NOT a third
axiom.**  It is the **abelianized shadow** of the correctly-stated flavoured
orthonormality — the R-valued pairing `I_{a,b} = Σ_w c_w(𝖖)·χ_w` has **every**
character coefficient `c_w(𝖖) ∈ Z[[𝖖]]` (non-negative 𝖖-powers, the flavoured
"no negative powers of 𝖖"; physically each flavour sector counts states with
`𝖖^{Δ+spin}`, `Δ,spin ≥ 0`), with the trivial-character diagonal `δ_{a,b}+O(𝖖)`.
Abelianizing each `χ_w` into μ-monomials spreads that one per-character
positivity into the coupled `|m| ≤ k+c` cone (Weyl + parity are the same
book-keeping).  With it imposed as the trace's *type*, the certificate
closes exactly: solution space = canonical trace × the **typed
rescale tower** `{μ^j 𝖖^i : i, j even, |j| ≤ i}` (Weyl-symmetrized) —
in particular **𝖖-odd rescales are excluded** by the typing
(spin-statistics of the index), unlike the trivial-R `1 + q·Q[[q]]`
line.  u1/su2u1 entries are refused pending the Plan-18 Z-form
regeneration (label-ρ² drops the μ^δ unit, so label-level cyclicity
is μ-violated — verified against the su(2)_{−4/3} closed forms).

Findings from the proof phase: (a) **e6's cyclicity-forced seed
relations** — twisted cyclicity imposes `T₂ = T₆` between distinct
ρ²-orbit seeds (satisfied exactly by the frozen characters) and
`T₀ = T₂ + 𝖖²T₄`.  `T₂ = T₆` is axiom 5 (ρ-equivariance of the trace,
`Tr∘ρ = Tr` at trivial flavour) on the ρ-paired seeds 2↔6; the other
ρ-paired seeds satisfy `T₅ = T₇` too (not forced by degree-1 cyclicity).
Seeds 4 and 5 lie in different ρ-orbits, so `T₄ ≠ T₅` is expected and says
nothing about ρ — the former "finer than `Tr∘ρ = Tr`, which is FALSE on e6"
reading was a mislabelled comparison (corrected 2026-09-18, `repo_audit.md` A73).
Pentagon/heptagon have zero such relations.  (b) **the e8 Layer-1
wall** — the tagged-cycle reducer fails to terminate (200k-step cap)
on degree-4 e8 cone monomials: the first observed boundary of the
"so-far empirical" Layer-1 universality; the e8 certificate therefore
uses pairs whose products stay at degree ≤ 3 (sound: any certifying
pair subset is a proof — under-constraining can only inflate the
dimension, never fake a certificate).  (c) **the window/N coupling** —
smaller N can require LARGER windows (truncation invalidates equation
orders): heptagon N=4 fails at deg 2 while N=6 certifies there.
All-orders (N = ∞) proofs need the order-k constraint-matrix
stabilisation — the Goal-2.11 trace-miracle phenomenon — the research
continuation of this program.  Next in the one-by-one queue: the
SU(2)-flavoured pair (a1d3, a1d5; character-valued seeds), then the
u1 series once the Plan-18 regeneration restores their reduction.

**Systematic continuation:** sweep the trivial-R zoo (e6: 7 seeds)
with the two-discriminator methodology; v2 Gram-primary runs as the
default instrument; flavoured entries after the Plan-18 Z-form
regeneration (the harness is trivial-R by construction until then);
beyond finite type, the v2 system is the candidate trace-uniqueness
instrument for pure U(N) and the quiver compositions (Plan 22).

## 6. Pointers

Code: `finite_kalgebras/` (`elem_traces`, `objects`, `regen`),
`kalgebra_object.py`, `rgkalgebra_object.py`, `bps_chart_object.py`,
`trace_uniqueness.py`, `implementations/finite_*_kalg.py`,
`implementations/a1d4_from_su3ad.py`.
Tests: `test_ade_serving_guard` (the per-row guard of §9), `test_finite_zoo_traces` (41), `test_kalgebra_object` (14),
`test_rgkalgebra_object` (19), `test_bps_chart_object` (9),
`test_su3_to_su2u1_hom` (15), `test_trace_uniqueness`.
Plans: 18 (Z-form regeneration — the gate for everything flavoured),
25 (object layer — landed), 26 (RG mutation — research).

---

## 7. Situation update (2026-06-11, curator reconciliation after #431–#442)

*(This section reconciles the map above with the twelve PRs that landed
after its writing; per-entry detail in Plan 29 notes + the inventory in
this section's sources.)*

**Proof state (Plan 29's math-proof target, first results):**

- **Pentagon: COMPLETE PROOF** that the standalone is a K_𝖖-algebra —
  all axioms K₁–K₄ including all-pairs orthonormality
  (`pentagon_kq_proof.md`, 959 lines; machine certificate
  `scripts/verify_pentagon_kq_proof.py`, 12 exact checks).
- **Heptagon / A₁[A₂ₖ]: K₁–K₃ proven** (k=2,3 exact; general-k
  spanning/termination); K₄ mapped — certified Andrews–Gordon tower
  closed forms; open frontier = mixed-long-chord monomials
  (`a1a2k_kq_proof.md` §7.4; #442 added the two-tail trace law).
- **Trace uniqueness** (the bootstrap program, Goal 2.12):
  pentagon/heptagon/nonagon proven to q⁶, e6 to q⁴ (pinned in CI);
  **e8 = partial certificate 14/17 seeds, PARKED ON A10** (three
  deep-charge seeds behind measured 8–15 GB memory walls; all 31 271
  axiom rows imposed; claim projected onto available-seed cells).
  Engines: `trace_uniqueness.py` (v1), `_flavoured.py` (v2/v3 Gram
  bootstrap — works without Layer-1/cones, extends to infinite type),
  `_proofs.py` (sparse exact solver, ~300× speedups).

**Coverage scoreboard:** standalones 14/14; frozen traces 9/14
(missing: e7/e8 compute-gated, a1d4/a1d6/a1d8 Plan-18-gated su2u1);
*(stale twice over by 2026-09-22 — e8 had been frozen, so 10/14 — and
superseded on 2026-09-23, when every table but e8's was removed in favour of
the live routes: see the status note at the top of `elem_traces.py`)*;
battery-clean natives: pentagon, heptagon, e6, a1d3, a1d5, a1d7;
u1 entries (a3/a5/a7) cured via `FiniteU1ZKAlgebra` Z-form wraps
(a3 full battery 0/0; a5 algebra+δ-trace verified; a7 algebra
exhaustive, trace layer the one open verification).

**Known-defect ledger (explicit, all with cures identified):**
1. a3 trapezoid defect — CURED (character refroze from
   `ad_characters.py`, K=48).
2. a5/a1d3/a1d5 latent tail defect — flavoured frozen tables exact
   only for |μ| ≲ 2(K−k)/3 (windowed `rg_generator` root cause;
   regenerate with margin or from closed forms).
3. su2u1 Z-forms (a1d4/6/8) — Plan 18 blocking item.
4. e6 structural finding: cyclicity forces T₂ = T₆ — which is axiom 5
   (ρ-equivariance of the trace) on the ρ-paired seeds 2↔6; T₅ = T₇ holds
   too; T₄ ≠ T₅ compares different ρ-orbits (corrected 2026-09-18) —
   pinned in tests.

**Coherence verdict (duplication scan): CLEAN.** No competing
implementations; `ELEM_TRACE_DATA` is the single trace source;
the two Z-form wrappers (u1, su2) are complementary instances of the
Plan-18 D1 pattern. The "spiraling" was **ledger lag**, not
incoherence — work landed faster than the coordination docs absorbed
it. Standing rule from this update: **every finite-stack PR appends
one line to this section** (what changed, where the proof/coverage
state moved).

**Adjacent architecture (#441):** Plan 30 (`AbeKAlgebra` contract
completion — the third presentation tier beside RGKAlgebra/TKAlgebra)
+ Plan 22 T3 linear-quiver layer (`QuiverOverPure`, `QuiverURQTorus`)
— the finite types remain the certified small-node oracles for both.

**Ledger 2026-06-14 (non-abelian trace bootstrap).** Extended the BPS-free
orthonormality bootstrap to the **su2** D-series: `su2_bootstrap.generate_su2`
(SU(2)-irrep forward sweep, Clebsch–Gordan-aware) closes a1d3/a1d5/a1d7
BPS-free (`Tr(1)` the only BPS call) — a1d3 (K≥12, 2/2), a1d5 (K≥10, 4/4),
a1d7 (K=6, 6/6) reproduce their frozen tables exactly; wired into
`elem_traces.generate`/`_seed_series`.
The **su2u1** Gram/window bootstrap is built (`su2u1_bootstrap` +
`experiments/su2u1_gram_probe.py`) and **diagnosed the true blocker**: it is
*upstream*, not the method.  The a1d4 standalone's **ρ is broken** (6/9
`verify_rho_is_automorphism` fail — the documented SU3AD-inherited defect; a1d8
4/36 + no `RHO_DELTA`), so orthonormality `I=Tr(ρ·)=δ+O(𝖖)` is violated on the
standalone and no trace bootstrap can close them until ρ is repaired (Plan-18 /
SU3AD fix).  **a1d6 has a clean ρ (0/36) + `RHO_DELTA`** → the one bootstrappable
su2u1 entry (gated on a scalable solver for its 39 mult-gens).  The
Gram-bootstrap's inconsistency is a *correct* ρ-axiom integrity signal.  Full
audit: `finite_type_trace_layer_audit.md`. Coverage delta: su2 trace layer now
BPS-free-derivable + cross-checked; su2u1 split into (clean) a1d6 vs
(ρ-defective) a1d4/a1d8.

**Ledger 2026-06-15 (su2u1 ρ repaired).** The su2u1 ρ defect is fixed (user:
"repair ρ first"; framing: *ρ permutes canonical labels — section ↦ (section, r)*).
New `finite_su2u1_zform.FiniteSU2U1ZKAlgebra` gives su2u1 labels the
`(SU(2) char, (cone word, U(1) charge))` structure (free=SU(2) ⊗ label=U(1),
`ρ=(⋆_su2, perm+δ−k)`); `experiments/compute_su2u1_rho_delta.py` recomputes the
U(1) shift `r` from the BPS σ.  **a1d8** `RHO_DELTA` (78/120) computed +
persisted → Z-form ρ 0/676; **a1d6** Z-form ρ 0/625; **a1d4** is a
section-perm-unclosed truncation (header flagged) — its ρ-correct realisation is
`a1d4_from_su3ad` (SU3ADKAlg, 0/16).  Test `tests/test_finite_su2u1_zform.py`.
Next: close the su2u1 *traces* on the now-ρ-clean Z-form surface.

## 8. The `[A_2, A_k]` family, and where it leaves the corner (2026-08-30)

The Argyres–Douglas family whose BPS quiver is the `A_2 ⊠ A_k` product the
quiver catalogue draws as its "rectangular form" (the `2 × k` bipartite grid,
every square cyclically oriented).  Rank `2k`.  Measured, with the catalogue's
own `[A_2,A_2] = D_4` and `[A_2,A_3] = E_6` as the controls:

| family | rank | cluster type | `dim ker B` | zoo id | in this corner? |
|---|---|---|---|---|---|
| `[A_2,A_1]` | 2 | `A_2` (pentagon) | 0 | `pentagon` | yes |
| `[A_2,A_2]` | 4 | `D_4` | 2 | `a1d4` | yes |
| `[A_2,A_3]` | 6 | `E_6` | 0 | `e6` | yes |
| `[A_2,A_4]` | 8 | **`E_8`** | 0 | `e8` | yes |
| `[A_2,A_k]`, `k ≥ 5` | `2k` | **not finite type** | 2 iff `3∣(k+1)` | — | **no** |

So the family exits the finite-type corner exactly at `k = 5`, and the four that
stay are — pleasingly — three of the corner's own entries plus `a1d4`.
`[A_2,A_2] = [A_1,D_4]` (the repo's `ad_voa_m5_paper.tex` says so too), which is
why the family's `k = 2` member is in the zoo under the `a1d4` name rather than
a `d4` one.

**`[A_2,A_4]` is `E_8`, not `E_7`** — `E_7` is not a square product of two
type-A quivers at all.  The old attribution is retired; see `repo_audit.md` A60.

**Order of ρ on the cluster variables** follows the corner's usual pattern:
measured 5, 4, 14, 16 for `k = 1..4` (and 10 for the neighbouring `E_7`), i.e.
`h + 2`, halved exactly where `−w₀ = id` (`D_4`, `E_7`, `E_8`).  These match the
σ-periods the catalogue already records.  (A reading valid in this `[A_2,A_k]`
corner only — `[A_1,D_{2n+1}]` has order `2n+1 = (h+2)/2` with `−w₀ ≠ id`, e.g.
the `Z_3` of `a1d3_kalg.py` at `D_3 = A_3`.  The user's `K_𝖖-algebras` draft,
Table `tab:finite_type`, lists all nine `[A_1,ADE]` families with their flavour
symmetry and order of ρ, and agrees with every order measured here —
kalgebra.md "Cross-reference with the `K_𝖖-algebras` draft", 2026-09-21.)

**What does *not* stop at `k = 4`.** A **finite BPS chamber** does not: the
`2 × k` grid chart has a negating sequence of length **`3k`** at every `k`
measured (`k = 1..6`, so 3, 6, 9, 12, 15, 18) — including `k = 5, 6`, which are
not of finite cluster type.  Finite chamber and finite cluster type are
independent; pure SU(2) is the standing example of the first without the second.
Consequently `[A_2,A_5]` has finite defining data — `(B, node charges, spec)` =
(10×10, 10, 15) — from which `BPSKAlgebra` derives the whole contract, even
though its cluster fan does not close (94 → 179 → 311 distinct monomial rays at
`build_cluster_graph_truncated` caps 500 → 2000 → 6000, `truncated` throughout,
against `k = 3` closing at 42 rays / 833 clusters).

**The tower.** `A_2 ⊠ A_{k−1}` is *literally* the induced subquiver of
`A_2 ⊠ A_k` on the first `2(k−1)` nodes, so deleting one end column is a
well-posed `SubquiverRG` at every `k`, and the rungs compose with `.then()` down
to the pentagon.  The flavoured members (`3∣(k+1)`) are built **u(1)²-gauged**,
following the repo's treatment of the flavoured members of `[A_1, A_odd]` and
`[A_1, D_even]`; then every rung is unflavoured and none carries a spectator
flavour into its IR.  The `[A_2,A_3]` rung lands on the standalone gauged
`[A_2,A_2]`, certified by a `KAlgebraIso`.  Module + suite:
`implementations/a2ak_induction.py`, `tests/test_a2ak_induction.py`.

### 8b. Cone presentation of the gauged members (2026-08-31)

Gauging is also what makes the cluster-cone machinery work on the flavoured
members.  On the u(1)^2-gauged `[A_2,A_2]` chart `build_cluster_graph_truncated`
closes naturally at **50 clusters / 16 monomial rays** — `D_4`'s cluster count
and its 16 almost-positive roots — where the *ungauged* chart returns `1 / 0`
with `truncated=False` (`repo_audit.md` A61).  So the gauged presentation is the
one that reproduces the honest cluster combinatorics, and the zoo's `a1d4`
(8/12) and the `D_4`-star chart (5/3) are flavour-quotient artifacts.

What does **not** yet work is emitting a `ConeKAlgebra` for a gauged member.
`generate_finite_kalg` offers only the four flavour patterns, and its cone stage
folds every direction orthogonal to the node charges into the lineality basis —
but for a gauged member those directions are the **invertible electric
generators**, which belong in `Cone(..., torus_gens=...)`, the kind
`U1A1AoddKAlg` marks by hand.  Pushed through the existing paths the gauged
`[A_2,A_2]` fails its own axioms (rho-automorphism 24/36 at `flavor="trivial"`,
26/36 at `"u1"`; bar 32/36; associativity 62/64) while the unflavoured
`[A_2,A_3]` control through the same pipeline passes 36/36, 36/36, 64/64.  A
torus-cone emitter path is the open task; `repo_audit.md` A63 records it.

## 9. The classes serving the ADE rows (Plan 42 R19), measured

Plan 42 ruling R19 (user, 2026-09-23) asks, for each ADE finite-type
`K_𝖖`-algebra — and its `U(1)`-gauged version where the flavour has a `U(1)` —
for *"self-contained realizations which are fully functional (arbitrary in
principle precision and coverage)"* (the user's words), and, for the `A` and `D`
families, geometric canonical-basis labels.  Read here (decisions of 2026-09-24,
listed for the user's veto): no frozen trace data and no BPS engine or RG flow
on the serving path; an orthonormality bootstrap seeded by the class's own
closed-form `Tr(1)` — exact, to any order, from the class's own data and the
orthonormality axiom — counts as self-contained and is recorded, not failed
(closed forms remain the preferred route where they are cheap); stored PRODUCT
data (cone or ρ tables, Python literals or a declared data file) is declared,
not failed — `frozen` is about trace data.  The table lists each class
whose methods form the serving path of such a row, as measured in a fresh
process by `tests/test_ade_serving_guard.py` under the instruments of
`restructuring_plans/42_paper_claims_battery/frozen_data_audit.py`.  The test's
checks, by name: `frozen` (no frozen trace data read: the import guard on the
frozen trace-table module, and no data file opened), `stored` (the modules on the
path dense in numeric literals are exactly the declared ones, listed below),
`bps`, `imports` (no bootstrap or RG-flow module), `bootstrap` (no
orthonormality-bootstrap routine runs), `tr1`, `gens`, `ortho` (`δ + O(𝖖)` on
generator pairs), `geometry`.  "Status" names the checks a row fails; each such
failure is listed, with its reason, in the test's `EXPECTED_FAILURES`, and the
test fails if any other check fails or a listed one passes.  The witnesses
(BPS charts, flows, retired classes) certify a row in its own tests and are not
on its serving path.

Windows: `Tr(1)` through `𝖖⁴⁰` and every multiplicative generator's trace
through `𝖖¹²` on every row — the even-D rows included since 2026-09-24, when the
closed forms of their seeds (`u1a1deven_seed_characters`, measured; Plan 42
`finite_type_trace_machinery.md` section J) took the generator traces off the
RG transport, and the Layer-1 reduction onto those seeds then took every other
trace, so every pairing, off it the same day (before, `A1DevenKAlg(3)` stopped
at `𝖖⁸`, the zoo's `a1d6` / `a1d8` were checked in a narrow window, and
`A1DevenKAlg(2)` / `(3)` sat behind `--slow`).  Orthonormality: all ordered
generator pairs where a row has at most 30 generators, else all pairs among
every `s`-th generator (`s = ⌈n/30⌉`) plus every diagonal pair; `--slow` takes
every ordered pair, except `A1DevenKAlg(3)` and the zoo's `a1d6` / `a1d8`,
which take the pairs of 60 generators (every pair at `a1d6`; 3,660 of the
14,400 at `A1DevenKAlg(3)` and at `a1d8`).  Measured 2026-09-24 on a shared
four-core machine, two processes: the default run (the controls and all 47
rows) in 259 s (0 failing); the six even-D class rows in 94 s
(`A1DevenKAlg(3)`'s 990 pairs 84 s) and with `--slow` in 188 s (`A1DevenKAlg(3)`'s
3,660 pairs 176 s; at the default sample's rate, all 14,400 would take about
20 minutes under the profile hook — an estimate, not run); the zoo's `a1d6` / `a1d8` with `--slow` in 28 s / 126 s.
The whole `--slow` run, re-timed 2026-09-24 on `main` at `cb0dc0bd` (same
machine, two processes): 47 rows, 0 failing, in 718 s — before these changes it
took 1.9 hours, of which `A1DevenKAlg(2)` and `A1DevenKAlg(3)` took 1.5 and 1.7.

| theory | serving class (module) | geometric labels (accessor) | products | traces (route) | stored data on the serving path | status | witness |
|---|---|---|---|---|---|---|---|
| `[A₁,A₂ₖ]`, `k = 1..3` | `A1A2kKAlg(k)` (`implementations/a1a2k_kalg.py`) | `curve(x, ell)`, `geometric_label(label)`: the diagonals of the `(2k+3)`-gon; ρ the rotation | cone reducer over the base table built at construction from chord geometry (`A1A2k_plucker_closed_form`) | Layer 1 + the `M(2,2k+3)` Andrews–Gordon characters | none | pass | the BPS chart `A1A2k(k).A` |
| `[A₁,A₂]` | `PentagonKAlg` (`kalgebra_samples.py`) | `curve(x, ell)`, `geometric_label(label)`: `L_i` is the diagonal `{3i, 3i + 2}` (the basepoint `L_0 = {0, 2}` a convention; certified against `A1A2kKAlg(1)`); ρ the rotation | cone reducer over `PENTAGON_CONE_DATA` | Layer 1 + the Rogers–Ramanujan closed forms | none | pass | `BPSKAlgebra` pentagon (a `KAlgebraIso`) |
| `[A₁,A₄]` | `HeptagonKAlg` (`kalgebra_samples.py`) | `curve(x, ell)`, `geometric_label(label)`: the diagonals of the heptagon (`(1, i)` = `{i, i + 2}`, `(2, i)` = `{i, i + 4}`; `A1A2kKAlg(2)`'s tuples with the orbit-2 index shifted); ρ the rotation | `A1A2kKAlg(2)`, relabelled | Layer 1 + `A1A2kKAlg(2)`'s `M(2,7)` seeds | none | pass | `heptagon_kalg.HeptagonKAlg` (BPS) |
| `U(1)`-gauged `[A₁,A₂ₖ₊₁]`, `k = 1..3` | `U1A1AoddKAlg(k)` (`implementations/u1a1aodd_kalg.py`) | `geometric_label(letter)`, `cone_data().chord`: the diagonals of the `(2k+4)`-gon | analytic peel (the even family's arc rules, rank-2 torus pairing) | Layer 1 + the `u1_pgon_layer2` closed forms | none | pass | retired `legacy/u1a1aodd_kalg_frozen.py`; the RG transport down `U1A1AoddToEvenQTRGKAlgebra`; `u1aodd_trace_bootstrap` |
| `U(1)`-gauged `[A₁,A₃]` | `U1HexagonKAlg` (`implementations/u1_hexagon_kalg.py`) | `geometric_label(letter)`: the diagonals of the hexagon, `U1A1AoddKAlg(1)`'s (`E^{±1}` unlabelled); ρ the rotation up to a power of `E` | `U1A1AoddKAlg(1)` through `(F, e) ↦ (F, −e)` | `U1A1AoddKAlg(1)`'s | none | pass | the frozen `FULL_PLUCKER_TABLE` (`cone_algebra.py`, off the path) |
| `[A₁,A₂ₖ₊₁]`, `k = 1..3` | `ungauge_u1a1aodd(k)`, an `UngaugedKAlgebra` (`implementations/ungauge_kalgebra.py`) | `geometric_label(label)`: balanced multisets of diagonals of the `(2k+4)`-gon (R20) | `U1A1AoddKAlg(k)` on the centralizer of `E` | the measure-restored gauge-charge sum of `U1A1AoddKAlg(k)`'s traces | none | pass | `A1AoddToEvenRGKAlgebra(1)` (`k = 1`); `u1_bootstrap.generate_u1`; the zoo's `a3` / `a5` / `a7` through their Z-form wrapper, a `KAlgebraIso` (`aodd_seeds.kalgebra_iso`, R22) |
| `[A₁,A₃]` | `HexagonKAlg` (`implementations/hexagon_kalg.py`) | `geometric_label(label)`: balanced multisets of diagonals of the hexagon (R20), `ungauge_u1a1aodd(1)`'s at `(F, −e)` | `U1HexagonKAlg` at magnetic charge 0 | the ungauging of `U1HexagonKAlg` | none | pass | — |
| `[A₁,A₅]`, `[A₁,A₇]`, `[A₁,A₉]` | `OctagonKAlg`, `DecagonKAlg`, `DodecagonKAlg` = `UngaugedPolygonKAlg(2, 3, 4)` (`implementations/ungauged_polygon_kalg.py`) | `geometric_label(label)` (R20) | `U1A1AoddKAlg(k)` on the centralizer | the ungauging of `U1A1AoddKAlg(k)` | none | pass | the retired `U1OctagonKAlg` / `U1DecagonKAlg` / `U1DodecagonKAlg` (`legacy/`) |
| `[A₁,D₂ₖ₊₃]`, `k = 0..2` | `A1DnKAlg(n)`, `n = 3, 5, 7` (`implementations/a1dn_kalg.py`) | `curve(x, ell, kappa)`: the curves of the once-punctured `n`-gon; ρ the rotation | `A1DoddConeKAlg((n−3)/2)` through the curve dictionary | Layer 1 + the `a1dodd_layer2` closed forms | none | pass | `a1dn_a1dodd_iso` |
| `[A₁,D₂ₖ₊₃]`, `k = 0..2` | `A1DoddConeKAlg(k)` (`implementations/a1dodd_kalg.py`) | `geometric_label(label)`: `A1DnKAlg(2k+3)`'s `(curves, κ)` of the same element, through its curve dictionary; ρ the rotation | the closed-form Ptolemy cone data `a1dodd_cone_data(k)` ⊗ SU(2) Clebsch–Gordan | Layer 1 + the `a1dodd_layer2` closed forms | none | pass | `a1dodd_trace_bootstrap` (through `𝖖⁴⁰` at `k = 1, 2`); the zoo's `a1d3` / `a1d5` / `a1d7` through their Z-form wrapper `FiniteSU2ZKAlgebra`, a `KAlgebraIso` (`a1dodd_seeds.kalgebra_iso`, R22) |
| `U(1)`-gauged `[A₁,D₂ₖ₊₂]`, `k = 1..3` | `U1A1DevenConeKAlgebra(k)` (`implementations/u1a1deven_cone_kalgebra.py`) | `curve(x, ell, e, kappa)`, `geometric_label(letter)`: the curves of the once-punctured `(2k+2)`-gon (`E^{±1}` unlabelled); ρ the rotation up to a power of `E` (R21) | the curve frame's closed-form product rule (`u1a1deven_geometric_frame`, the `A1Dodd` arc rules) | the gauge sector from Creutzig's closed form; the seeds — an odd curve, or a non-crossing pair of a +1 and a −1 curve, times `Eⁿ` — from their closed forms (`u1a1deven_seed_characters`, measured, since 2026-09-24); every other label by the cone data's Layer-1 reduction onto the seeds, and the pairing by multiply-then-trace (since 2026-09-24) | none | pass | the transport route (`seed_closed_forms=False`: the transport on the closed-form RG image into `A1DoddConeKAlg(k−1) ⊗ QT(Z²)`, built without the flow); the flow `U1A1DevenViaDoddRG(k)`; the retired table class (`legacy/u1a1deven_cone_kalgebra_tables.py`) |
| `[A₁,D₂ₖ₊₂]`, `k = 1..3` | `A1DevenKAlg(k)` (`implementations/a1deven_kalg.py`) | `geometric_label(label)`, inherited from `UngaugedKAlgebra` over the gauged class's letter hook: balanced multisets of curves of the once-punctured `(2k+2)`-gon (R20, R21) | `U1A1DevenConeKAlgebra(k)` on the centralizer, the gauge charge moved into the `U(1)` fugacity | the ungauging of `U1A1DevenConeKAlgebra(k)`: every generator from the seeds' closed forms, products (so the pairings) by its Layer-1 reduction onto them, to any order | none | pass | `A1DevenRGKAlgebra(k)`; the gauged class's transport route; the zoo's `a1d4` / `a1d6` / `a1d8` through their Z-form wrapper, a `KAlgebraIso` (`a1deven_seeds.kalgebra_iso`, R22) |
| `[A₁,D₄]` (SU(3) flavour) | `SU3ADKAlg` (`implementations/su3_ad_kalg.py`) | `geometric_label(label)`: the curves of `A1DevenKAlg(1)` — balanced multisets of curves of the once-punctured square (R20), `T_i` the loop at `i` with the curve `(i + 1, 2)`, `D_i` the curve `(i + 1, 3)` — with the SU(3) weight; ρ the rotation, the weight conjugated (since 2026-09-24) | the hard-coded `Z₄` relations of the module | Layer 1; `Tr_1`, `Tr_T`, `Tr_D` from the even-D `k = 1` closed forms (`u1a1deven_seed_characters`: Creutzig's gauge tower for `Tr_1`, the seeds through the curve map for `Tr_T`, `Tr_D`), summed over the gauge charge with the measure restored (`sl3_su3_traces`, since 2026-09-24) | none | pass | `SU3BPSKAlgebra`; the Kac–Wakimoto vacuum of `sl(3)_{−3/2}` (the `Tr_1` route until 2026-09-24; equal through `𝖖¹⁰⁰`) and the forward orthonormality pass `SU3ElemTraces` (the `Tr_T`, `Tr_D` route; equal through `𝖖⁸⁰`); `A1DevenKAlg(1)` through the map |
| zoo `pentagon`, `heptagon` | `FINITE_KALGEBRAS[id]` (`implementations/finite_pentagon_kalg.py`, `finite_heptagon_kalg.py`) | `geometric_label(label)`: the diagonals of the pentagon / heptagon, `A1A2kKAlg(1)` / `A1A2kKAlg(2)`'s (`finite_kalgebras.zoo_geometry`) | the exported cone table | Layer 1; `Tr(1)` and the seeds through `A1A2kKAlg(1)` / `A1A2kKAlg(2)` (`finite_kalgebras.aeven_seeds`, since 2026-09-24): exact to any order | the module's cone table (1.4 / 5.7 literals per line) | pass | the trivial-R bootstrap `elem_traces._generate_bootstrap` over the `M(2,5)` / `M(2,7)` vacuum character (the route until 2026-09-24); the BPS engine on request (`elem_traces._bps_oracle`), for every zoo id |
| zoo `a3`/`hexagon`, `a5`/`octagon`, `a7`/`decagon` | `FINITE_KALGEBRAS[id]` (`finite_a3_kalg.py`, `finite_a5_kalg.py`, `finite_a7_kalg.py`) | `geometric_label(label)`: balanced multisets of diagonals of the `(2k+4)`-gon (R20), `ungauge_u1a1aodd(k)`'s (`zoo_geometry`) | the exported cone table | Layer 1; the seeds through `ungauge_u1a1aodd(k)` (`finite_kalgebras.aodd_seeds`) | the module's cone table | pass | `ad_characters.a3_elem_entry`; `u1_bootstrap.generate_u1`; the seeds map as a `KAlgebraIso` from the Z-form wrapper `FiniteU1ZKAlgebra` (base-changed along `μ ↦ z⁻¹`) onto `ungauge_u1a1aodd(k)` (`aodd_seeds.kalgebra_iso`, R22) |
| zoo `a1d3` | `FINITE_KALGEBRAS["a1d3"]` (`finite_a1d3_kalg.py`) | `geometric_label(label)`: curves of the once-punctured triangle, `A1DoddConeKAlg(0)`'s (`zoo_geometry`) | the exported cone table | the seeds through `A1DoddConeKAlg(0)` (`finite_kalgebras.a1d3_seeds`) | the module's cone table | pass | `su2_bootstrap.generate_su2`; the seeds map as a `KAlgebraIso` from the Z-form wrapper onto `A1DoddConeKAlg(0)` (`a1dodd_seeds.kalgebra_iso`, R22) |
| zoo `a1d4` | `FINITE_KALGEBRAS["a1d4"]` (`finite_a1d4_kalg.py`) | `geometric_label(label)`: balanced multisets of curves of the once-punctured square (R20), `SU3ADKAlg`'s (`zoo_geometry`) | the exported cone table | the seeds through `SU3ADKAlg` restricted to SU(2)×U(1) (`finite_kalgebras.a1d4_seeds`), so from the even-D `k = 1` closed forms since 2026-09-24 | the module's cone table | pass | `A1DevenKAlg(1)` (`finite_kalgebras.a1deven_seeds`' `k = 1` map, a `KAlgebraIso` from the Z-form wrapper `FiniteSU2U1ZKAlgebra` since R22, `a1deven_seeds.kalgebra_iso`); the zoo quiver's BPS chart |
| zoo `a1d5`, `a1d7` | `FINITE_KALGEBRAS[id]` (`finite_a1d5_kalg.py`, `finite_a1d7_kalg.py`) | `geometric_label(label)`: curves of the once-punctured pentagon / heptagon, `A1DoddConeKAlg(1)` / `A1DoddConeKAlg(2)`'s through `finite_kalgebras.a1dodd_seeds` (`zoo_geometry`) | the exported cone table | the `a1d5_layer2` / `a1d7_layer2` closed forms | the module's cone table | pass | `A1DoddConeKAlg(1)` / `A1DoddConeKAlg(2)` through `finite_kalgebras.a1dodd_seeds`: `Tr(1)` and every generator's trace equal the images' (the general-`k` `a1dodd_layer2`), through `𝖖¹⁶` / `𝖖¹²`; the map as a `KAlgebraIso` from the Z-form wrapper `FiniteSU2ZKAlgebra` (`a1dodd_seeds.kalgebra_iso`, R22) |
| zoo `a1d6`, `a1d8` | `FINITE_KALGEBRAS[id]` (`finite_a1d6_kalg.py`, `finite_a1d8_kalg.py`) | `geometric_label(label)`: balanced multisets of curves of the once-punctured hexagon / octagon (R20, R21), `A1DevenKAlg(2)` / `A1DevenKAlg(3)`'s (`zoo_geometry`) | the exported cone table | the seeds through `A1DevenKAlg(2)` / `A1DevenKAlg(3)` (`finite_kalgebras.a1deven_seeds`, since 2026-09-24): every seed and `Tr(1)` to any order (the seeds' closed forms); products by Layer 1 over the seeds | the module's cone table | pass | the flow `A1DevenRGKAlgebra(k)`; the seeds map as a `KAlgebraIso` from the Z-form wrapper `FiniteSU2U1ZKAlgebra` onto `A1DevenKAlg(2)` / `A1DevenKAlg(3)` (`a1deven_seeds.kalgebra_iso`, R22) |
| `[A₁,E₆]`, `[A₁,E₈]` | `FiniteE6KAlgebra`, `FiniteE8KAlgebra` = zoo `e6`, `e8` (`finite_e6_kalg.py`, `finite_e8_kalg.py`) | — | the exported cone tables | Layer 1 + the `W₃(3,7)` / `W₃(3,8)` character recipes (`finite_kalgebras.w3_seeds`) | the modules' cone tables (12.3 / 207 literals per line) | pass | `E₆`: the Nahm sum + orthonormality bootstrap; `E₈`: the former `K = 6` table (a fixture of `tests/test_w3_seeds.py`) |
| `[A₁,E₇]` | `FiniteE7KAlgebra` = zoo `e7` (`finite_e7_kalg.py`) | — | the exported cone table | Layer 1 + the theta-product recipes and the Bershadsky–Polyakov vacuum (`finite_kalgebras.e7_seeds`) | the module's cone table; `e7_seeds.py`'s recipe coefficients (1.3 literals per line) | pass | the corrected `u1_bootstrap.generate_u1` |
| `U(1)`-gauged `[A₁,E₇]` | `U1E7ConeKAlgebra` (`implementations/u1e7_cone_kalgebra.py`) | — | cone reducer over `u1e7_cone_tables.pkl` and `u1e7_rho_tables.pkl`, built once on the flow `U1E7GaugedRG` | magnetic charge `≠ 0` ⇒ 0; the neutral sector through `FiniteE7KAlgebra` | the two pickles (product and ρ tables, declared); `_E7_GENERATOR_PREIMAGES` (the dictionary to `FiniteE7KAlgebra`); the `E₇` cone table; `e7_seeds.py` | pass | the flow `U1E7GaugedRG` (`U1A1E7RGKAlgebra`) |

Layer 1 runs on canonical labels (`ConeData.layer1_on_labels`; Plan 42
`finite_type_trace_machinery.md` section O) for the zoo's `e8`, `e7`, `a5`,
`a7`, `a1d4`, `a1d6` and `a1d8`; the other zoo rows keep the reduction on words.

What the checks say.  Since 2026-09-24 no row fails any check.  `geometry`
passes on every `A` and `D` row: every `A` and `D` theory has a family class
with geometric labels (`A1A2kKAlg`, `U1A1AoddKAlg`, `ungauge_u1a1aodd` and the
named ungauged polygons, `A1DnKAlg`; for even `D`, `U1A1DevenConeKAlgebra` and
`A1DevenKAlg`, R21; for `[A₁,D₄]` with its SU(3) flavour, `SU3ADKAlg`, on
`A1DevenKAlg(1)`'s curves), and every other presentation reads its labels
through its certified map onto one — the named samples `PentagonKAlg`,
`HeptagonKAlg`, `U1HexagonKAlg`, `HexagonKAlg` and the cone frame
`A1DoddConeKAlg` (#1579), and the zoo's entries through
`finite_kalgebras.zoo_geometry`, each by the map that serves its traces
(`a1d5` / `a1d7`, whose traces are their own closed forms, by the map of
`finite_kalgebras.a1dodd_seeds`, found for the purpose and certified on every
generator product and on ρ).  A map is fixed only up to the family's rotation
ρ, an automorphism, so a presentation's labels are those of its served map.  `bootstrap` would be
recorded, not a defect (the reading of "self-contained" above), but since
2026-09-24 no row pins seeds by orthonormality at run time: the zoo's pentagon
and heptagon seeds (pinned by `_generate_bootstrap` until then) come from the
closed-form `A1A2kKAlg(k)`, and `SU3ADKAlg`'s `Tr_T`, `Tr_D` (a strictly
triangular forward pass until then, which the zoo's `a1d4` inherited) from the
even-D `k = 1` closed forms through its curve map; the two bootstraps stay as
witnesses.
`imports` and `frozen` pass on every row: the even-`D` rows, and the zoo's `a1d6`
/ `a1d8` served through them, since the curve frame became the public class
(R21: no RG module on the serving path, no pickle); `U1E7ConeKAlgebra` declares
its two pickles (product and ρ tables; its traces are closed forms) and passes.
The test's `EXPECTED_FAILURES` is empty.

The seeds maps as `KAlgebraIso`s (R22, 2026-09-26).  For the flavoured zoo
entries (`a3` / `a5` / `a7`, `a1d3` / `a1d5` / `a1d7`, `a1d4` / `a1d6` /
`a1d8`), the generator maps of `aodd_seeds`, `a1dodd_seeds` and
`a1deven_seeds` are now `KAlgebraIso`s (`kalgebra_iso(short_id)` in each).
Each runs from the zoo's Z-form wrapper, which moves the flavour from the
coefficients into the labels, onto the family class.  The battery passes on
every generator and flavour character, every ordered pair of them, ρ and the
traces through `𝖖¹²` (at `a1d8` on a documented sample by default, in full
with `--slow`).  `finite_kalgebras.objects.kalgebra_object` holds the wrapper
and the family class as a component of their own ('z-form' → the family key).
No `KAlgebraIso` joins that component to 'cone-frozen', which keeps its
flavour in its coefficients.  The `[A₁,A₂ₖ]` entries already had theirs: the
object layer's 'cone-frozen' → 'a1a2k' witness is `aeven_seeds`' served map
itself.  Record and numbers: Plan 42 `finite_type_trace_machinery.md`
section N.
