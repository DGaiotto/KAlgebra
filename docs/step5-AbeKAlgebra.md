# AbeKAlgebra — the abelianized-presentation tier for `A_𝖖[T]`

**Step 5**, the abelianized layer of this repository (`src/abe/`). This is the tier
that presents a K_𝖖-algebra **faithfully on an enriched rational quantum torus** —
the abelianized (gauge-fugacity / difference-operator) description of the
K-theoretic Coulomb branch algebras of conventional gauge theories: pure U(N) and
SU(N), U(N) with `N_f` fundamentals, and linear quivers. Those algebras are built by
the general-gauge-group classes of Step 7 (`PureGAbeKAlgebra`, `GNAbeKAlgebra`, both
`AbeKAlgebra` subclasses); this layer holds the tier's contract, its torus
substrate, the native `PureSU2KAlgebra`, and the object layer. It is also the *capstone*
that ties the presentations together: every algebra built here is certified
against the BPS / cone / RG presentation of the *same* abstract algebra by a
`KAlgebraIso`, and the object layer holds those presentations under one roof.

## This layer relies on Steps 1–4 (by design)

Steps 1–3 are deliberately spine-free. **Step 5 is the opposite: it depends on the
earlier layers.** The abelianized presentation of a *matter* theory is precisely a
pure-gauge `AbeKAlgebra` combined with an `RGKAlgebra` flow that
adjoins the matter; and the certification web uses the cone (Step 2), RG (Step 3),
and BPS (Step 4) presentations as the other endpoints of its isomorphisms. Every
module imports the earlier layers by bare name — nothing is duplicated (importing a
Step-5 module without the earlier layers on the path raises `ModuleNotFoundError`).
Because it imports the BPS spine, its self-test runs **last** in the gate, after the
spine-freeness assertions of Steps 1–3. Six of its object and isomorphism modules
also import Step 7, which now builds the U(N) algebras they wrap, and the pure SU(2)
object imports Step 6 for its skein presentation.

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
`ρ`/`ρ⁻¹` (in closed form on labels, `wrq_torus.rho_label`, with the torus G-twist
read back as the verifier `verify_rho_via_twist`; linear quiver chains still read
the twist back), `trace`/`inner_product` (the Schur-measure residue pairing), the
chart-level bar verifier, and `certify_canonical` (the executable acceptance test:
bar-palindromicity together with the single-leading q-extreme condition). Charts
are memoized, and the memo persists: `save_cache` / `load_cache`, with a
presentation fingerprint and every loaded element re-verified.

**The constructive-build rule** is contract-enforced: the tier exposes no solve
entry point. A canonical is assembled constructively, as a polynomial in
already-built bar-invariant generators (dressed minuscules, `det`, Wilson lines),
never solved for — a solve can land *outside* the canonical span, where both
orthonormality and the internal span-pairing test would pass on off-span content.
Every decomposition is certified by exact reconstruction. The one licensed solve —
the (★)-guarded solve of `star_bubbling`, which the Step-7 classes build with — lives
outside the tier, in one module that applies the whole guard before it returns
anything: (★) asks that a line's difference operator preserve the Neumann module
`R(G)`, a membership condition that orthonormality cannot see.

## The seven blocks

**A — pure U(N).** Pure U(N) is `PureGAbeKAlgebra(u_n(N))` (Step 7), its
canonical basis labelled by the 't Hooft–Wilson charge `(m, e)` and every element
built by the (★)-guarded solve. The block runs the tier's contract on it at U(2):
the `AbeKAlgebra` base's derived `multiply` and `ρ` agree with the class's own; the
bar involution, the ρ-automorphism property and orthonormality
`I_{a,b} = δ_{a,b} + O(𝖖)` hold on a window of labels; and `certify_canonical`
accepts each of them. Substrate: the group-general `WRQTorus` (`root_datum`,
`weyl_torus_ring`, `wrq_torus`).

**B — matter.** U(N) with fundamental matter is `GNAbeKAlgebra` (Step 7), with named
presets in `g_matter_roster`; the block checks orthonormality of `roster('u2-nf1')`
(U(2) with one fundamental, coefficient ring `UNZPlusRing(1)`) on a window of
magnetic, electric and flavour labels. Linear quivers are `GNAbeKAlgebra` on a
product datum with bifundamental matter.

**C — N=2\*.** `SU2N2StarRGKAlgebra` is the SU(2) adjoint-matter RG flow: a live
`RGKAlgebra` over `R(U(1))` whose `S_RG` carries even characters.

**D — the object layer.** `pure_u2_object()` / `u2_nf1_object()` return
`KAlgebraObject`s holding the abe, cone (`QTCone`), and BPS
presentations of one abstract algebra, wired with certified `KAlgebraIso`
witnesses (`abe ↔ cone` identity-on-labels, full battery; `abe ↔ bps` the chamber map to
charges `γ`) and `verify_coherence`, a path-independence certificate across all
three; the `abe` presentation is `PureGAbeKAlgebra(u_n(2))`, and for U(2)+1 the
`roster('u2-nf1')` preset. `pure_un_bps_iso(N)` is the standalone
`PureGAbeKAlgebra(u_n(N)) ≅ BPS pure U(N)` witness.

**E — the SU(2) family.** SU(2) is non-minuscule, so `PureSU2KAlgebra` is a
genuinely native `AbeKAlgebra` (a native peel + bar-correction) — the one
type-specific class the tier keeps, a construction of pure SU(2) that does not go
through the (★)-guarded solve and is much faster on deep dressed labels.
`AbelianizedSU2KAlg` realises pure SU(2) as the trace-zero subalgebra of pure U(2).
The `KAlgebraObject`s tie these to the other layers: `pure_su2_object` holds the
`abe`, `bps`, `cone` and `skein` presentations; `su2_nf1_object` holds `cone`,
`bps`, `abe` (`SU2Nf1FlavourInCoefficients`, over `roster('su2-nf1')`) and
`bps-rform`, and additionally carries **RG-flow legs** (decouple the matter dyon →
pure SU(2); decouple the 't Hooft node → the A₂ pentagon); `su2_unf_object` holds
`cone` and `bps` (its abelianized presentation with the `SO(2N_f)` index
enhancement is not carried in this release).

**F — SU(3) / U(3).** Pure SU(3) is `PureGAbeKAlgebra(su_n(3))`, orthonormal on a
window of labels; pure U(3) is certified `≅` its BPS realisation
(`pure_un_bps_iso(3)`).

**G — the flow-typed object layer.** `RGKAlgebraObject` refines `KAlgebraObject` to
flows: a witness is an `RGKAlgebraIso` — the UV-side iso plus an auxiliary iso, with
flow verifiers (`verify_rg_intertwine` / `verify_s_rg_match` / `verify_apex_match`).
`subquiver_flow_object` presents one BPS node-deletion flow in two descriptions (an
F-oracle `SubquiverRG` vs a bare co-solver) bridged by a flow witness;
`su2a1d3_rg_object` / `su2a1d4_rg_object` are the concrete SU(2)-gauged `[A₁,Dₙ]`
flow objects, including an Abe + RG object whose IR auxiliary factors through the
abelianized pure SU(2).

## What's included (`src/abe/`)

The tier and its substrate: `abe_kalgebra.py`, the rational-quantum-torus engines
(`wrq_torus`, `matter_wrq_torus`, `quiver_wrq_torus`, and the `urq_torus` family —
`urq_torus`, `matter_urq_torus`, `quiver_urq_torus` — kept as the substrate of the
vacuum-state pairing's URQ form; `qtorus_1d`, `abelianized_torus`,
`weyl_torus_ring`), and the Lie-theoretic data (`root_datum`, `pure_ade`,
`pure_ade_lattice`, `sun_cartan_reduction`, `so2nf_characters`).

Pure gauge: `pure_su2_kalgebra`, `abelianized_su2_kalgebra`, `un_bps_chamber`; the
U(N) and SU(N) algebras are Step 7's `PureGAbeKAlgebra`.

Matter: `su2_nf2_kalgebra`, `su2_n2star_rgkalgebra`, `su2_unf_object`; U(N) with
matter and linear quivers are Step 7's `GNAbeKAlgebra`.

The object layer: `pure_u2_object`, `u2_nf1_object`, `pure_su2_object`,
`su2_nf1_object`, `su2_unf_object`, and the flow-typed `su2a1d3_rg_object`,
`su2a1d4_rg_object`; the isomorphism witnesses (`pure_un_bps_iso`,
`abelianized_su2_bps_iso`, `pure_su2_bps_iso`, `pentagon_bps_iso`, `su2_nf1_h_iso`,
`su2_nf2_h_iso`, `iso_composed_rgkalgebra`, `rgkalgebra_iso_via_ir`); the BPS
decoders it certifies against (`bps_su2_nf1/2/3`, `su2_nf1_bps_decoder`,
`su2_nf1_bps_rform`, `pure_u2_qtcone`, `u2_nf1_qtcone`); and the atlases of the
gauged and quiver families (`su2_family_atlas`, `su2_gauged_a1dn_atlas`,
`su3_family_atlas`, `su3_quiver_atlas`, `pure_su3_atlas`).

## Tests

```bash
python3 run_tests.py        # the full gate; test_abe_flows.py runs last
```

`tests/test_abe_flows.py` exercises all seven blocks: the tier's contract and
orthonormality on pure U(2) (A), matter orthonormality on `roster('u2-nf1')` (B), the
SU(2) N=2\* flow (C), the object layer with certified `KAlgebraIso`s and coherence
(D), the SU(2) family with its RG-flow legs, its skein presentation and coherence
(E), pure SU(3) orthonormality and U(3) `≅` BPS (F), and the flow-typed
`RGKAlgebraObject`s (G).

## License

GPL-3.0-or-later (see `LICENSE`).
