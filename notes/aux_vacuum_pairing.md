# The vacuum-state formulation of the Schur pairing — `I_{a,b} = ⟨L_a·1, L_b·1⟩` on the rational quantum torus

**Charter (user, 2026-07-24).** Sessions cannot reliably answer questions about
the left/right action structure on wavefunctions (half-indices, 3d indices) —
the code and documentation are ambiguous (the deformation lives in a *measure*
on one page, a *twist* on another, a *bra* on a third).  Fix: *"define the
non-Abelian structures in a way which is compatible with how Abelian
calculations are done.  So one would have an inner product on the rational
quantum torus algebra elements `Σ_m f_m(𝖖^m v) U_m`, lift it to formal sums like
`f_m(ζ) = ∏_α (ζ^α 𝖖²;𝖖²)_∞ = |1⟩` with good properties under left- and right-
multiplication by L's and define `I_{a,b}` as inner products of `L_a|1⟩` and
`L_b|1⟩`"* — with the expectation *"the pairing to involve a m-Levi Vandermonde
for v, but has to be tested"*.  Ultimately this definition is to be used
**universally in code and documents** (user ruling, same session).

**Implementation: `aux_space.py` (root) — `AuxSpace`; battery `tests/test_aux_space.py`.**
Plan: `restructuring_plans/40_aux_vacuum_pairing/`.  Probes (all rerunnable):
`experiments/aux_vacuum_pairing_{u2,u3_measures,u3_regular,exact_ct}.py`.

---

> **Boundary/half-index consumers (2026-08-05).**  The `I_{a,b}` vacuum-state
> pairing below remains THE primary definition of the Schur pairing (user
> ruling 2026-07-24).  For BOUNDARY states — Neumann of pure/`(G,N)` bulks,
> their 3d-chiral enrichments, and two-sided interfaces — the production
> surface is now `implementations/enriched_neumann_boundary.py` /
> `enriched_neumann_interface.py` (the spherical frame: vacuum reciprocal in
> the bra, `F^N_B = 1/[ψ_B·Z_B]`-type states, both commuting actions), whose
> dictionary to this document's aux frame is measured in
> `experiments/neumann_wavefunction_actions.py` [retired 2026-09-19 with the type-A keystone; the measurement stands as recorded] (Parts B0/C/E).

## 1. The definition (validated scope: see §3)

> **⚠ THE WEIGHT `w_m` BELOW IS SUPERSEDED — read §4 for the live form.**  This
> section records the definition as first written, with the weight in the shape
> the charter above *expected* it ("the pairing to involve a m-Levi Vandermonde
> for `v`, but **has to be tested**" — user, 2026-07-24).  It was tested, and the
> weight is now **derived rather than chosen** (D1 RESOLVED; §4):
>
>     w_m(v) = ∏_{α>0} (−𝖖)^{−s_α} (1−𝖖^{s_α}v^α)(1−𝖖^{s_α}v^{−α}) · R_{−m,m}(𝖖^m v)
>
> with `s_α := |⟨m,α⟩|` — the charge-shifted **full** Vandermonde times the
> `U_{−m}U_m` cocycle at the half-shifted argument, uniform in `s_α` *including*
> `s_α = 0`, and root-datum-general.
>
> **The difference is substantive, not presentational**, which is why this banner
> exists rather than a footnote: the m-Levi form below is *trivial at regular `m`*,
> where the derived form is not.  The two agree at central `m` (both reduce to the
> full Vandermonde) — so a reader who takes §1 as current gets the right answer on
> exactly the sector that would not reveal the error.
>
> Everything else in this section — the vacuum `|1⟩`, states as plain torus
> products, the three pairings by number of dressed slots, the Abelian
> degeneration — stands unchanged.  `aux_space.py` and CLAUDE.md carry the §4
> form; this section is the historical record of how it was reached.

Fix a pure U(N) theory on the `URQTorus` (repo frame: stored residuals, atoms
`U_m`).  Define, as *bona fide torus elements* (truncated formal sums):

    |1⟩  :=  (𝖖²;𝖖²)_∞^N · ∏_α (𝖖² v^α; 𝖖²)_∞ · U_0        (α over all roots)

    L_a|1⟩  :=  chart(a) · |1⟩         (plain URQTorus LEFT multiplication)
    |1⟩·L_a  =  right multiplication    (= the ρ-image left action; §6e)

and the **bare pairing** (no ρ anywhere, no deformed weight):

    ⟨x, y⟩  :=  (1/N!) Σ_m ∮ ∏_i dv_i/(2πi v_i) · w_m(v) · x_m(1/v) · y_m(v)

with `w_m` the **m-Levi Vandermonde**

    w_m(v)  =  ∏_{α : ⟨m,α⟩ = 0} (1 − v^α)        [full Vandermonde at m central,
                                                    trivial at m regular]

evaluated on the *stored residuals at bare `v`* (no `𝖖^{±m/2}` evaluation
shifts — the storage frame IS the wavefunction frame; measured, §3).  Then

    I_{a,b}  =  Tr(ρ(a)·b)  =  ⟨L_a·1, L_b·1⟩            (2 dressed slots)
    (a|Π]    =  ⟨L_a·1, Π⟩                                (half-index: 1 dressed slot)
    (Π|Π')   =  ⟨Π, Π'⟩                                   (3d index: 0 dressed slots)

**One bare measure; one `P`-dressing per state slot; all 𝖖-deformation lives in
the states.**  The old `(1|`-bra ("`P` is the measure"), the §6b per-`m` weight
`W_m`, and the ρ-cocycle `_pairing_f0` route are equivalent *computations*; this
is the intended primary *definition* (docs pivot = Plan 40 T4).  It also
resolves a live terminology trap: "Schur measure" is used for **two different
objects** — the half-index integrand `Δ^conv·P` (one `P`, from one dressed slot)
and the trace/Gram integrand `Δ^conv·P²·(𝖖²;𝖖²)^{2N}` (two dressed slots).

**Abelian compatibility (the charter's point).**  At U(1)ⁿ the root set is
empty: `P = 1`, `w_m = 1`, and the pairing degenerates to
`Σ_m ∮ x(m, 1/v)·y(m, v)` — literally the `threed_index_operators.DiffOp`
picture (`X(a,b)·f(m,e) = 𝖖^{am−be} f(m−b,e−a)`, `apply_left`/`apply_right`,
right = ρ-image of left).  Non-Abelian = same structure + nontrivial `P` and
Weyl measure.  No new mechanism.

## 2. What ρ-freeness means here

The Hermitian slot (left argument at `1/v`) plus the two vacuum dressings
*generate* the ρ-twist: `⟨a·b, c⟩ = ⟨b, c·ρ⁻¹(a)⟩` (left-mult adjoint =
right-mult by `ρ⁻¹`) holds with `ρ` never constructed — the certified engine's
`G̃_n` √measure cocycle is replaced by conj + `P`-in-state.  Verified against
`tests/test_boundary_aux_inner_product.py`'s adjointness discrimination
(`ρ² ≠ id` on dyonic pure-U(N) labels, so `ρ` vs `ρ⁻¹` is a real distinction).

## 3. Test record (2026-07-24)

**U(2), numeric grid, two 𝖖-values (`aux_vacuum_pairing_u2.py`; the script was retired with the URQ keystone on 2026-09-19 — the measurement stands, and `tests/test_aux_space.py` now certifies the same pairing on the WRQ form against `PureGAbeKAlgebra(u_n(N))`).**  10/10 pairs
match `PureUNKAlgebra.inner_product` (certified `== Tr(ρ(a)b)`): identity,
Wilson `(0,0),(1,0)`, 't Hooft `(1,0)`, **dyonic** `((1,0),(1,0))` (Witten
monodromy, ρ²≠id), det `(1,1)`, all cross-orthogonality zeros.  The convention
scan collapsed to a **unique** combination: Levi measure (charge-shifted and
full-Vandermonde candidates fail), bare-`v` frame (`t=0`; `t=±1` fail), global
constant `= (𝖖²;𝖖²)_∞^{2N}/N!` — matched as a *prediction* at U(3).

**U(3), numeric grid, partial Levis (`aux_vacuum_pairing_u3_measures.py`).**
The discriminating test the U(2) run cannot do (rank-1 Levis are full or
trivial).  All single-canonical pairs — including 't Hooft `(1,0,0)` and
`(1,1,0)` (Levi `U(1)×U(2)` / `U(2)×U(1)`) and the dyonic — match under the
m-Levi measure at both 𝖖-values to ~1e-5, with the predicted constant; `full`
(×4.6), `triv` (×0.45), charge-shifted `mag` (×1.4) all fail.  **The user's
m-Levi expectation is confirmed at every sector with broken-root levels
`|⟨m,α⟩| ≤ 1` and at central sectors.**

**Exact q-series confirmation (`aux_vacuum_pairing_exact_ct.py`).**  Replacing
the grid by exact constant-term extraction:

| orbit | bare (×N!) | certified (×N!) |
|---|---|---|
| `H1` orbit `(1,0,0)` | `6 −18𝖖² +12𝖖⁴ +12𝖖⁸ +6𝖖¹⁰ −66𝖖¹² +54𝖖¹⁴` | **identical** |
| `L_(2,1,0)` bubbling orbit `(1,1,1)` | `6𝖖⁴ −12𝖖⁶ −18𝖖⁸ +36𝖖¹⁰ +48𝖖¹² −96𝖖¹⁴` | **identical** |
| `L_(2,1,0)` regular orbit `(2,1,0)` | `12 −48𝖖² +60𝖖⁴ +0𝖖⁶ −84𝖖⁸ +…` | `6 −24𝖖² +30𝖖⁴ −12𝖖⁶ −6𝖖⁸ +…` |

(the certified per-orbit series are extracted from the engine's own
`_pairing_f0` per-sector sum, Weyl-orbit-grouped — sector-diagonality makes
them frame-invariant and directly comparable.)

## 4. Sectors with `|⟨m,α⟩| ≥ 2` — the window law (RESOLVED at `|s|=2`; general `|s|` open)

> **Retraction (same session, later).**  The first version of this section
> claimed the per-slot window factorization was *refuted* by exact division.
> That refutation was an artifact of a **bugged division routine** (it peeled
> the extremal monomial toward which the binomial shifts, so it could never
> terminate and always reported "not divisible").  With the division done
> correctly, the factorization **holds exactly** — see below.  The measure
> candidates that were genuinely tested and genuinely fail: charge-shifted
> full Vandermonde `∏(1−𝖖^{|⟨m,α⟩|}v^α)` (fails already at `|s|=1`),
> GNO/inter-level *measure* windows (overshoots ×~6), level-`𝖖²` factor at
> `|s|=2`.  Only the division-based refutation was wrong.

**The confirmed law (`|s|=2`).**  At a sector `m` containing a broken pair
with `|m_i − m_j| = 2`, the state residual of `chart·|1⟩` factors **exactly**
(as a full q-series) as

    F_m(v)  =  (1 − v^α) · G_m(v),          α the difference-2 pair root,

and the pairing is the m-Levi-Vandermonde CT of the **reduced** wavefunctions
`G_m` — equivalently, the weight divides by the window norm
`(1−v^α)(1−v^{−α})`, whose poles are exactly cancelled by the states' window
zeros.  Verified **exact through 𝖖¹⁴**, per Weyl orbit:

- U(3), `L_(2,1,0)` regular orbit: reduced pairing `==` certified
  `= 6 −24𝖖² +30𝖖⁴ −12𝖖⁶ −6𝖖⁸ +72𝖖¹⁰ −144𝖖¹² +12𝖖¹⁴` (×N!);
- U(2), `L_(2,0)`, *both* orbits: reduced `(2,0)`-orbit
  `2 −8𝖖² +6𝖖⁴ +16𝖖⁶ −22𝖖⁸ −16𝖖¹⁰ +10𝖖¹² +48𝖖¹⁴` `==` certified; the
  bubbling `(1,1)` orbit bare-central `==` certified.

This also explains the once-mysterious leading factor 2 in the table above:
the `𝖖⁰`-slice of `F_m` is exactly `1 − v^α`, and
`∮(1−v^α)(1−v^{−α}) = 2` versus the wanted `1` per sector.

**THE DERIVED CLOSED FORM (this session, principled route — supersedes the
guessing above).**  Per the user's direction, the weight was *derived* by
expressing the certified §6b pairing in the vacuum-state language: §6b gives
`⟨f,g⟩_m = ∮ W_m^{sym}·f(1/v)·g(v)` with `W_m^{sym} = μ(𝖖^m v)·K_m(𝖖^m v)`,
`K_m = v̄(ψ_m)·mono_m·S_{−m}(ψ_m)` (all closed forms: `ψ_m` §2, `mono_m` §5,
half-shift sampling §6b), and the vacuum contributes `P(𝖖^m v)` per slot.  So

    w_m  =  μ(𝖖^m v)·K_m(𝖖^m v) / [ P(𝖖^m/v)·P(𝖖^m v) ]

— a pure factor-multiset cancellation (every ingredient is a binomial
`(1−𝖖^e v^{±α})` or an infinite tail of them; ALL tails cancel — verified —
leaving a finite product).  Result, per pair `(i,j)` with `d = |m_i−m_j|`
and `x = v_i/v_j`:

    d = 0 :   (1−x)(1−x⁻¹)                    (undeformed Levi Vandermonde)
    d ≥ 1 :   ∏_{k=1}^{d−1}  1 / [ (1−𝖖^{d−2k}x)(1−𝖖^{d−2k}x⁻¹) ]

i.e. **the m-Levi Vandermonde divided by the squared interior window** —
levels `d−2, d−4, …, 2−d` symmetric about zero (empty at `d=1`; the single
level-0 factor at `d=2`, reproducing the division law; levels `±1` at `d=3`,
whose `(1−𝖖⁻¹x)` explains the once-mysterious `𝖖⁻²` symptom and `1+x²`
slice).  Where the states carry matching zeros (`d=2`) the weight poles are
cancelled and the "divide the window out of each slot" form is an equivalent
presentation; at `d ≥ 3` the weight poles sit off the unit contour (levels
`≠ 0`) and the geometric expansion is unambiguous.

Derivation script: `experiments/aux_vacuum_pairing_derive_weight.py`
(U(2), `m=(d,0)`, `d = 0..4`; conventions engine-anchored — note `ψ`'s
`S = Σ_j j·m_j` and `mono`'s `2Σ_t t·k_t` are both **0-indexed** in the code,
`_psi_dom` / `_rho_block_data`).  **Verified**: `d = 0,1,2` reproduce all
earlier exact confirmations; `d = 3` verified exact on `L_(3,0)` (both Weyl
orbits, through the trusted window) — `experiments/aux_vacuum_pairing_d3_check.py` [retired 2026-09-19 with the type-A keystone; the measurement stands as recorded].

**Structural identification (user, 2026-07-24): `w_m` is the interior of the
`U_{−m}U_m` cocycle.**  The sector-`m` contribution to `Tr(ρ(a)b)` reaches the
magnetic-0 residue through `U_{−m}·U_m = R_{−m,m}(v)·U_0`,
`R_{−m,m} = ψ_{−m}·S_{−m}(ψ_m)`, whose divisor is the W-boson cone of `m`
(poles at `v_i = 𝖖^{2l}v_j`, `l = 0..d`; multiplicity 1 at the edges, 2 at
the interior — §6a).  In the vacuum-state frame the vacuum tails + measure
zeros absorb the edges and reduce `μ` to the undeformed Levi Vandermonde on
unbroken pairs; what survives of `R_{−m,m}` is exactly its interior double
poles, re-centered by the half-shift (`l = 1..d−1 ↦` levels `2l−d`).  So
`w_m = (Levi Vandermonde) / (interior of the U_{−m}U_m cocycle)` — the pairing
of sector `m` with itself collides a monopole/anti-monopole pair and the
weight retains only the strictly intermediate screened W-boson levels; the
boundary levels are carried by the states' vacuum dressing.  (Edge
cancellation visible in the raw derivation output at `d=2`: the numerator
`(1−𝖖⁻²x⁻¹) = −𝖖⁻²x⁻¹(1−𝖖²x)` exactly kills the edge factor.)

**The exact cocycle form (user's identification, verified).**  Per broken
pair (`x = v_i/v_j`, `d = |m_i−m_j|`), the weight is exactly

    w_m  =  (−1)^{d+1} (x + x⁻¹ − 𝖖^d − 𝖖^{−d}) · R_{−m,m}(𝖖^m v) ,

i.e. **(the magnetic Vandermonde) × (the `U_{−m}U_m` cocycle at the
half-shifted argument)** — `(v⁻¹−v)(ṽ⁻¹−ṽ)` with `v = 𝖖^{−d/2}ζ`,
`ṽ = 𝖖^{d/2}ζ` expands to precisely `x⁻¹ − 𝖖^d − 𝖖^{−d} + x`.  The formula
is uniform in `d` **including `d = 0`** (where `R` is trivial and
`−(x+x⁻¹−2) = (1−x)(1−x⁻¹)` is the Levi factor): the "Levi Vandermonde ÷
interior windows" form above is this identity with the cocycle poles made
explicit.  The `(−1)^{d+1}` sign echoes the paper's `(−1)^B` Weyl-twist for
odd 't Hooft charges (SU(2)/SO(3) subtlety) — its preferred absorption is a
presentation choice.  Verified by exact factor algebra at U(2), `d ≤ 4`
(`w_d/R_{−m,m}(𝖖^m v)` computed with the derivation machinery).

**Final datum-general form (user's push to eliminate "interior";
verified d ≤ 4).**  The same identity, written with the charge-shifted FULL
Vandermonde (no interior notion, no type-A combinatorics; s_α := |⟨m,α⟩|):

    w_m(v)  =  ∏_{α>0} (−𝖖)^{−s_α} (1−𝖖^{s_α}v^α)(1−𝖖^{s_α}v^{−α})
               · R_{−m,m}(𝖖^m v) ,

uniform in s_α INCLUDING s_α = 0.  Every ingredient is root-datum-general
(WRQTorus-ready): the shifted Vandermonde over all roots, the `U_{−m}U_m`
cocycle at the half-shifted argument, and the monomial prefactor — whose
exponent `Σ_{α>0}|⟨m,α⟩| = Δ(m)` is the GNO monopole dimension, so the
prefactor is `(−𝖖)^{−Δ(m)}`.  (The cocycle sits in the NUMERATOR
necessarily: w_m's poles share R's orientation — cone denominators — and no
ψ-cocycle has cone zeros, so a `Vandermonde / R'` form does not exist except
as word-play with inverse atoms.)  Note the shifted full Vandermonde is the
very first scan's failed standalone candidate ('mag') — it was the correct
Vandermonde half of the weight, missing its cocycle factor.  General-G
verification (beyond the U(N)-derived ψ/R) belongs to T3's WRQTorus
certification.

**D1 status: derived, no longer a free choice.**  The only remaining D1 item
is presentational: state the definition with the weight in this
window-denominator form (recommended — uniform in `d`), with the `d=2`
per-slot-division corollary noted.  The atom-renormalization option is moot.

## 4m. Matter: the derived per-cell window Ξ (2026-07-25)

With cotangent matter split one-per-slot into the vacuum (ruling A4: the
ket gains the sign-half letter `E(μ x_w) = 1/((−𝖖 μ x_w); 𝖖²)_∞` per matter
cell of weight `w`, `a_n = nahm_term((−1)^n, n, [n])` by Euler; the bra slot
conjugates `v` and `μ`), the full weight is `w_m^gauge · ∏_cells Ξ_cell`
with, per cell (slot fugacity `μ`, gauge weight `w`, `c := ⟨m, w⟩`):

    Ξ_cell  =  1                                                (c ≥ 0)
    Ξ_cell  =  ∏_{s=0}^{|c|−1} (1 + 𝖖^{1−|c|+2s} μ⁻¹ v^{−w})
                               (1 + 𝖖^{1−|c|+2s} μ    v^{+w})   (c < 0)

**Derivation** (`experiments/aux_vacuum_pairing_matter_derive.py`): the same
factor-multiset cancellation as the gauge weight, per cell.  After the CT
substitution `v → 𝖖^m v`, the trace-side matter cells cancel the ket's
shifted letters EXACTLY; the matter cocycle collapses to plain
`W_{−m,m}(𝖖^m v) = Z_{−m}(v)·Z_m(v)` (since `T_{−m} ∘ (v→𝖖^m v) = id`); and
the ρ matter bookkeeping (Z-top monomial division + the level star, which
rungs only at `c > 0`) converts the leftover bra window into the Hermitian
double-window above — or to exactly 1 for `c ≥ 0`.  Consequences:

- **why the fundamental probes saw no matter window**: their test sectors
  only carried `c ≥ 0` cells — and `Ξ ≡ 1` there is now derived μ-refined,
  not just observed at fugacities → 1;
- the `c < 0` window is the paper's N=2* dressing-numerator structure
  `(𝖖 v_i + μ v_j)` (the `u_±` factors), appearing uniformly for adjoint,
  fundamental, and bifundamental cells;
- the first (μ→1) N=2* run failed at `⟨H1,H1⟩` with `q⁰ = 9` because the
  uncorrected integrand has an on-contour pole at μ→1: the truncated
  geometric tower summed to `WLEV+1 = 9`; the Ξ numerators are precisely
  the collapsing factors.

**Validation (all μ-refined, level-by-level vs the certified engine
traces):** N=2* U(2) — `|c| = 1` (H1) and `|c| = 2` (H1·H1, incl. the
`𝖖^{−1}` window binomials) (`aux_vacuum_pairing_n2star.py`); U(1)×U(1)
with one bifundamental link — the window is the ENTIRE weight there; ten
pairings incl. `c = ±1, ±2` and the other-node trigger
(`aux_vacuum_pairing_bifund.py`); N_f = 2 flavour-refined — per
flavour-level VECTOR against `MatterURQTorus.trace` (incl. an anti-'t Hooft
with `c = −1` flavour cells), and the SU(2)-weight marginal against
`UNNfKAlgebra.inner_product`'s χ-expansion (D8b central collapse)
(`aux_vacuum_pairing_matter_nf2_refined.py`).

Implementation note (truncation discipline): multiply each Ξ half-window
into ITS OWN slot's sector residual first — each factor telescopes its own
letter tower exactly, pushing the truncation remnant to level `WLEV+1` —
then cap slot levels where post-collapse level `n` costs `𝖖^{≳n}`.  The
`μ⁻¹`-half becomes the ket-form factor on the bra slot under the pairing
conjugation.  Promotion of the matter surface into `aux_space.py` is staged
(T5/T6); the experiments carry the reference implementation.

## 4g. General-G certification (2026-07-25)

`AuxSpace.from_datum(root_datum)` runs the identical definition on the
group-general WRQTorus substrate (roots as exponent vectors, `⟨m,α⟩` dot
products, `|W|` normalization).  **Certified at SU(2)** against
`PureSU2KAlgebra` (itself certified vs the BPS Kronecker chart):
identity / Wilsons / the basic 't Hooft / dyon / `L_{2,0}` / off-diagonals
(`tests/test_aux_space.py::test_su2_wrq_datum`,
`experiments/aux_vacuum_pairing_su2_wrq.py`).  SU(2) exercises what U(N)
cannot:

- `⟨m, α⟩ = 2m`: the basic 't Hooft already opens the `s = 2` level-0
  window, and `L_{2,0}` opens `s = 4` (off-contour expansion levels ±2);
- the `m = 1` chart's magnetic-0 residual BUBBLES (rational): its
  denominators are cleared exactly against the vacuum zeros by q-shifted
  binomial division inside `_monos` — another genuineness certificate;
- `|W| = 2` with `dim = 1` decouples the normalization from `N!`/`N`.

Two measured cautions.  (i) *Internal q-headroom*: sectors with
`s_α ≥ 3` windows convolve expansion towers whose products transiently
exceed `𝖖^{K+2}`; `pair()` now adds per-sector headroom
`2·Σ_{α>0} max(0, s_α − 2)` (no step lowers q-order, so headroom only adds
true terms — at `s = 4`, `cap = K+2` left a spurious `−9·𝖖⁸`).  (ii) *No
per-atom-pair referee*: the referee trace's individual `(−μ, μ)` atom-pair
terms have on-contour poles and are NOT individually well-defined (measured:
the `L_{2,0}` pair terms trace to garbage/zero while the full sum is exact)
— only whole-element traces referee.

## 4g-bis. Beyond SU(2): `Spin(5)`, `Sp(2)`, `G₂` — **16/16 exact** (2026-07-27)

§4g certified `from_datum` at SU(2) only, because SU(2) was the only non-U(N)
datum with certified canonicals.  Since #1040/#1044 there are canonicals at
groups with **no minuscule cocharacter** (`Spin(5)`, `G₂`) and at `Sp(2)`, so the
pairing can now be checked where it was never calibrated.

| datum | pairs | result |
|---|---|---|
| `Spin(5) = B₂` | identity, Wilsons `ω₁`/`ω₂`, 't Hooft `((1,1),(0,0))`, dyon `((1,1),(1,0))`, 3 off-diagonals | **8/8 MATCH** |
| `Sp(2) = USp(4)` | identity, Wilson, 't Hooft, cross | **4/4 MATCH** |
| `G₂` | identity, Wilson, 't Hooft `((2,3),(0,0))` (175 s), 't Hooft×Wilson (40 s) | **4/4 MATCH** |

**Why this is a real cross-check and not a tautology.**  The two sides share the
`WRQTorus` substrate and nothing else.  `PureGAbeKAlgebra.inner_product` is the
§6b chart pairing — construct `ρ(L_a)` by the torus G-twist, multiply, take the
Schur-measure residue.  `AuxSpace.pair` is the vacuum-state pairing, which
**never constructs `ρ`**: the Hermitian slot plus the two vacuum dressings
generate the twist, with the derived weight `w_m`.  So agreement tests the
derived weight against the ρ-cocycle off its calibration set — **and**
independently exercises the (★)-solved canonicals, since a wrong bubbling tail
would break the pairing even while satisfying the chart-side identity.

What these data exercise that SU(2) and U(N) cannot: **non-simply-laced** root
systems, where `s_α = |⟨m,α⟩|` differs across `α` inside one sector so the
per-root window structure is not uniform; and `G₂` at `|W| = 12` with 6 positive
roots — the largest weight `w_m` the pairing has been asked for.

Battery row: `tests/test_aux_space.py::test_general_g_datum`.  Probe:
`experiments/aux_vacuum_pairing_general_g.py`.

## 5. Rulings already in place (user, 2026-07-24)

- **Placement**: torus-level, consumed by half-indices and 3d indices — a root
  module (`aux_space.py`, name provisional, D2) or grown in `urq_torus.py`;
  *not* AbeKAlgebra-tier-only.
- **Datum-general from the start** (roots from `RootDatum`/`TorusShape`;
  `WRQTorus`/SU(N) inherit).
- **Universal adoption**: this definition ultimately replaces the scattered
  presentations in code and documents (T4).
- **Matter**: the matter trace factors split one-per-slot into the vacuum
  (half-hyper letters); **non-cotangent matter out of scope** for now.

## 6. Cross-references

| topic | doc |
|---|---|
| the auxiliary space as currently documented (bra frame, `W_m`, ρ-cocycle) | `pun_enriched_qt.md` §6b/§6d/§6e |
| the wavefunction/difference-equation map (Abelian side) | `wavefunction_difference_equations.md` |
| left/right = `u`/`ũ`, adjointness tests | `tests/test_boundary_aux_inner_product.py` |
| the N=2* line operators `L_{m,e}` | `restructuring_plans/36_n2star_adjoint_matter/` |
| plan / decisions / tasks | `restructuring_plans/40_aux_vacuum_pairing/` |
