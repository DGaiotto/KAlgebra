# AbeKAlgebra — the abelianized-presentation tier for `A_𝖖[T]`

**Step 5**, the abelianized layer of this repository (`src/abe/`). This is the tier
that presents a K_𝖖-algebra **faithfully on an enriched rational quantum torus** —
the abelianized (gauge-fugacity / difference-operator) description of the
K-theoretic Coulomb branch algebras of conventional gauge theories: pure U(N) and
SU(N), U(N) with `N_f` fundamentals, and linear quivers. It is also the *capstone*
that ties the presentations together: every algebra built here is certified
against the BPS / cone / RG presentation of the *same* abstract algebra by a
`KAlgebraIso`, and the object layer holds those presentations under one roof.

## This layer relies on Steps 1–4 (by design)

Steps 1–3 are deliberately spine-free. **Step 5 is the opposite: it depends on the
earlier layers.** The abelianized presentation of a *matter* theory is precisely a
pure-gauge `AbeKAlgebra` (the keystone) combined with an `RGKAlgebra` flow that
adjoins the matter; and the certification web uses the cone (Step 2), RG (Step 3),
and BPS (Step 4) presentations as the other endpoints of its isomorphisms. Every
module imports the earlier layers by bare name — nothing is duplicated (importing a
Step-5 module without the earlier layers on the path raises `ModuleNotFoundError`).
Because it imports the BPS spine, its self-test runs **last** in the gate, after the
spine-freeness assertions of Steps 1–3.

## The tier — `abe_kalgebra.py`

`AbeKAlgebra(KAlgebra)` presents an algebra on the enriched rational quantum torus,
in the **f-presentation**

    x  =  Σ_{m⃗}  f_{m⃗}(𝖖^{m⃗} v) · U_{m⃗},

with `U_{m⃗}` the enriched atoms of the quiver rational quantum torus and `f_{m⃗}`
rational in the gauge fugacities `v` (and flavour fugacities `μ`). A realisation
supplies exactly three primitives —

  * `torus_shape()` — the shape of the enriched torus (per-node root data, matter
    multiplicities, links);
  * `chart(label)` — the faithful f-presentation image of the canonical `L_label`;
  * `decompose(x)` — the level-ascending canonical read of a torus element,
    which **raises off its certified scope rather than guessing** —

and the whole `KAlgebra` API is derived: `multiply = decompose(chart · chart)`,
`ρ`/`ρ⁻¹` (the torus G-twist read back), `trace`/`inner_product` (the Schur-measure
residue pairing), the chart-level bar verifier, and `certify_canonical` (the
executable acceptance test: bar-palindromicity together with the single-leading
q-extreme condition).

**The constructive-build rule** is contract-enforced: the tier exposes no solve
entry point. A canonical is assembled constructively, as a polynomial in
already-built bar-invariant generators (dressed minuscules, `det`, Wilson lines),
never solved for — a solve can land *outside* the canonical span, where both
orthonormality and the internal span-pairing test would pass on off-span content.
Every decomposition is certified by exact reconstruction. (The tier's *own*
surface is unchanged by Step 7's licensed solve: that licence lives in one
auditable module and carries the guard the ban presupposes absent —
[`step7-GNKAlgebra.md`](step7-GNKAlgebra.md).)

## The product law, and the cocycle that is primitive

Atoms multiply by a cocycle, and **the cocycle is the primitive** — defined by a
closed form, with everything else derived from it:

    U_m · U_{m'}  =  CC[N]_{m,m'} · U_{m+m'},        R = T_{a+b}(CC)

`CC = CC[0]` is the pure-gauge case, a genuine specialisation matching the
`(G, N)` / `(G, 0)` convention, and `CC[N]` is what the product law multiplies
by. Both charged sectors are products over the charged directions, and a
direction contributes exactly when the two magnetic charges pair with it in
**opposite signs**. Writing `A`, `B` for the two pairings, both factors live on
one Clebsch–Gordan range `d = ||A|−|B||` to `D = |A|+|B|` stepping by 2, and the
vector/matter parallel is exact: the vector multiplet spans the **closed** range
as denominators (ends once, interior twice), the matter spans its **strict
interior** as numerators (once each).

Two consequences worth having. `CC_{a,b} = 1` whenever `a`, `b` and `a+b` share a
closed Weyl chamber — all the content is chamber-crossing. And `CC` satisfies the
bar axiom at torus level, `bar(CC_{a,b}) = CC_{b,a}`, because every ingredient is
symmetric in `|A|, |B|` so the swap and the charge conjugation coincide. The
bare-argument `R` does **not** (measured: it fails on 28 of 81 charge pairs at
SU(3) where `CC` fails on none), and that is the second reason `CC` is the
primitive rather than a convenience.

Nothing in the closed form mentions the atom dressing, the atom phase or the
Weyl transport: it absorbs them all, including the half-integral-height case. So
the dressing `ψ` and the matter numerator `Z` are demoted to **trivializations**,
whose role is to let the (★) condition be stated — and since `R` and the matter
cocycle are now defined independently, `δψ = R` and `δZ = W` become *emergent*
evidence rather than definitions.

## `ρ` on labels, and the memo

`ρ`/`ρ⁻¹` are the **explicit label-level closed form**. On Weyl orbits of pairs,
with no chamber assumed,

    ρ^{±1}[(m, e)] = [( −m,  −e + Σ_{α∈Φ: ±⟨α,m⟩>0} |⟨α,m⟩|·α
                              − Σ_{w∈wt(N): ±⟨w,m⟩>0} |⟨w,m⟩|·w )]

— `ρ` uses the roots and matter weights *positive* on `m`, `ρ⁻¹` the negative
half; vector multiplet with `+`, hypermultiplet with `−`. At `N = Adj` the two
sums cancel identically and `ρ` is the antipode `[(m,e)] ↦ [(−m,−e)]`, with
`ρ² = id` on gauge charges. The former route — read the torus twist back through
`decompose` — is demoted to a **verifier**, which is the honest place for it.

The charts are memoized, and the memo **persists**: `save_cache(path)` /
`load_cache(path)`, the same names and shape as the RG tier's. A realisation opts
in by exposing its live `{label: element}` memo, and one file can carry a
delegated algebra's charts too. The file is exact — an integer numerator over an
explicit denominator multiset — so a round trip is an identity, not a
re-derivation. Two guards, because a cache file is untrusted input: the header
fingerprints the presentation (with the phase convention *probed*, since it is a
callable and cannot be compared otherwise) and refuses a mismatch, and every
admitted chart is re-verified against the axioms. Memoization itself is
deliberately **not** a switch — the memo is structural — but `clear_cache()`
reclaims it.

## The seven blocks

**A — the pure-U(N) keystone.** `PureUNKAlgebra(N)` labels its canonical basis by
the 't Hooft–Wilson charge `(m, λ)` (an anti-dominant cocharacter `m`, a dominant
Levi irrep `λ`); `ρ` is a sign-free permutation. Every canonical is a polynomial in
bar-invariant generators (dressed minuscules `L_E`/`L_F`, `det`, Wilson `L_W`),
hence bar-invariant by construction, with each decomposition certified by exact
reconstruction. Orthonormality `I_{a,b} = δ_{a,b} + O(𝖖)` is tested directly.
Substrate: the `URQTorus` closed-form engine and the group-general `WRQTorus`
transport (`root_datum`, `weyl_torus_ring`, `wrq_torus`).

**B — matter as Abe + RG.** `UNNfKAlgebra(N, N_f)` is the native abelianized
U(N)+N_f algebra (labels `(m, λ)` at N_f=1; `((m, λ), w)` over `R(SU(N_f))` for
N_f > 1 — the diagonal U(1) sits in the gauge centre, so the genuine flavour is
SU(N_f)). `UNNfOverPure` is the same algebra as an `RGKAlgebra` (Step 3) over
`PureUNKAlgebra.add_flavour(N_f)` with the matter spectrum generator. That these
present the same abstract algebra as the BPS-quiver realisation is certified by a
full `KAlgebraIso` battery at U(2)+N_f=1. Linear quivers: `UNQuiverKAlgebra` /
`QuiverOverPure` (per-node U(N_i) chains with bifundamental links).

**C — the N=2\* machinery.** `N2StarBuilder(N)` implements the physical recipe
"solve in N=2\*, then RG-flow": it builds a torus chart from magnetic + dressing
data, which the pure keystone `decompose`s to a single canonical — the
pure-U(N)/SU(N) `L_{m,e}` finder (minuscule / aligned-tower / adjoint sectors,
N=2,3,4). `SU2N2StarRGKAlgebra` is the corresponding SU(2) adjoint RG flow.

**D — the object layer.** `pure_u2_object()` / `u2_nf1_object()` return
`KAlgebraObject`s holding the abe (keystone), cone (`QTCone`), and BPS
presentations of one abstract algebra, wired with certified `KAlgebraIso`
witnesses (`abe ↔ cone` identity-on-`(m,λ)`, full battery; `abe ↔ bps` the
`(m,λ) ↔ γ` chamber map) and `verify_coherence`, a path-independence certificate
across all three. `pure_un_bps_iso(N)` is the standalone `PureUNKAlgebra(N) ≅ BPS
pure U(N)` witness.

**E — the SU(2) AbeKAlgebra family.** SU(2) is non-minuscule, so `PureSU2KAlgebra`
is a genuinely native `AbeKAlgebra` (a native peel + bar-correction), not a
projection of the U(2) keystone; `PureSUNKAlgebra` generalises it.
`AbelianizedSU2KAlg` realises pure SU(2) as the trace-zero subalgebra of the
pure-U(2) keystone; `SU2Nf1Abe` and `SU2UNf{N_f}AbeKAlgebra` carry the matter
cases (with the SO(2N_f) index enhancement — triality at N_f=4). The
`KAlgebraObject`s `pure_su2_object` / `su2_nf1_object` / `su2_unf_object` tie these
to the other layers; `su2_nf1_object` additionally carries **RG-flow legs**
(decouple the matter dyon → pure SU(2); decouple the 't Hooft node → the A₂
pentagon).

**F — SU(3) / U(3).** Pure U(3) is the keystone at N=3; pure SU(3) is the native
`PureSUNKAlgebra(3)`. Orthonormality holds for both, and each is certified `≅` its
BPS realisation (`pure_un_bps_iso(3)`, `pure_sun_bps_iso(3)`).

**G — the flow-typed object layer.** `RGKAlgebraObject` refines `KAlgebraObject` to
flows: a witness is an `RGKAlgebraIso` — the UV-side iso plus an auxiliary iso, with
flow verifiers (`verify_rg_intertwine` / `verify_s_rg_match` / `verify_apex_match`).
`subquiver_flow_object` presents one BPS node-deletion flow in two descriptions (an
F-oracle `SubquiverRG` vs a bare co-solver) bridged by a flow witness;
`su2a1d3_rg_object` / `su2a1d4_rg_object` are the concrete SU(2)-gauged `[A₁,Dₙ]`
flow objects, including an Abe + RG object whose IR auxiliary factors through the
abelianized pure SU(2).

## What's included (`src/abe/`)

The tier and its substrate: `abe_kalgebra.py` (the contract, and the persistable
chart memo), the rational-quantum-torus engines (`urq_torus`, `wrq_torus` — which
carries the primitive cocycle `CC` and the label-level `ρ` — `matter_urq_torus`,
`matter_wrq_torus`, `quiver_urq_torus`, `quiver_wrq_torus`, `qtorus_1d`,
`sun_rq_torus`, `abelianized_torus`, `weyl_torus_ring`), and the Lie-theoretic
data (`root_datum`,
`pure_ade`, `pure_ade_lattice`, `sun_cartan_reduction`, `so2nf_characters`).

Pure gauge: `pure_un_kalgebra`, `pure_sun_kalgebra`, `pure_su2_kalgebra`, plus the
construction/chart engines (`pure_un_construct`, `pure_un_canonical`,
`pure_un_chart_engine`, `pure_un_closed_form`, `un_bps_chamber`) and the N=2\*
finder (`pure_via_n2star`).

Matter and quivers: `un_nf_kalgebra`, `un_nf_dressed_generators`, `un_quiver_kalgebra`,
`un_nf_over_pure_rgflow`, `quiver_over_pure`, `su2_nf2_kalgebra`, `su2_n2star_rgkalgebra`,
and the SU(2) abelianized family (`abelianized_su2_kalgebra`, `su2_nf1_abe`,
`su2_unf_abe_kalgebra`, `su2_unf_object`).

The object layer: `pure_u2_object`, `u2_nf1_object`, `pure_su2_object`,
`su2_nf1_object`, and the flow-typed `su2a1d3_rg_object`, `su2a1d4_rg_object`; the
isomorphism witnesses (`pure_un_bps_iso`, `pure_sun_bps_iso`, `un_nf_bps_iso`,
`un_nf1_over_pure_iso`, `abelianized_su2_bps_iso`, `su2_nf1_h_iso`, `su2_nf2_h_iso`,
`iso_composed_rgkalgebra`, `rgkalgebra_iso_via_ir`); and the BPS decoders/objects it
certifies against (`bps_su2_nf1/2/3`, `su2_nf1_bps_decoder`, `su2_nf1_bps_rform`,
`pure_su2_object`, `pure_u2_object`, `u2_nf1_object`, `pure_u2_qtcone`,
`u2_nf1_qtcone`, and the per-theory object modules).

## Tests

```bash
python3 run_tests.py        # the full gate; test_abe_flows.py runs last
```

`tests/test_abe_flows.py` exercises all seven blocks plus the tier's own
substrate — the primitive cocycle's bar axiom (and the measured *failure* of the
same axiom for the derived `R`, without which that check would be vacuous), the
label-level `ρ` round trip on dominant representatives, the chart cache's exact
round trip and its fingerprint refusal, and `LaurentPoly`'s refusal of a
non-integral coefficient. The seven blocks: the keystone contract and
orthonormality (A), matter as native Abe and the flow ↔ BPS iso (B), the N=2\*
builder and SU(2) N=2\* flow (C), the object layer with certified `KAlgebraIso`s and
coherence (D), the SU(2) family with its RG-flow legs and coherence (E), SU(3)/U(3)
native algebras and their BPS isos (F), and the flow-typed `RGKAlgebraObject`s (G).

## License

GPL-3.0-or-later (see `LICENSE`).
