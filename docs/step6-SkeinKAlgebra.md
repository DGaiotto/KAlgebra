# SkeinKAlgebra — skein algebras of surfaces as `A_𝖖[T]`

**Step 6**, the skein layer of this repository (`src/skein/`). It realises the
**SU(2) (Kauffman-bracket) skein algebras of marked surfaces** as K-theoretic
Coulomb branch algebras `A_𝖖[T[A₁, Σ]]` — the line-defect Schur sector of the
4d 𝒩=2 theory of class S obtained from `Σ`. A closed curve on the surface
becomes an SU(2)-character (Wilson-type) canonical generator, an open arc a
monomial generator, an ideal triangulation a BPS-quiver chart, and a diagonal
flip a quiver mutation.

## This layer relies on Steps 1–5 (by design)

Steps 1–3 are deliberately spine-free; Steps 4–6 are the opposite. **Step 6
depends on every earlier layer.** It imports the core contract (Step 1), the
cone tier and its polygon / D-family catalogue (Step 2), the RG engine
(Step 3), the BPS spine (Step 4) and the abelianized tier's pure-gauge / BPS
decoders and object layer (Step 5) — all by bare name, nothing duplicated.
Importing a Step-6 module without the earlier layers on the path raises
`ModuleNotFoundError`: the dependency is real, not a copy. Because it imports the
BPS spine, its self-test runs **last** in the gate, after the spine-freeness
assertions of Steps 1–3.

## The dictionary (physics ↔ topology)

| skein | Coulomb branch |
|---|---|
| `Sk_{A₁}(Σ)` over `Z[A^{±1}]` | `A_𝖖[T[A₁, Σ]]` (the line-defect Schur sector) |
| simple closed curve | SU(2)-character canonical generator (Wilson-type) |
| open arc (marked boundary) | monomial canonical generator |
| Kauffman resolution of a crossing | structure constants of the canonical basis |
| ideal triangulation | BPS-quiver chart |
| diagonal flip | quiver mutation (certified `KAlgebraIso`) |
| quantum trace | the "geometric F" — the bare part of `F(−γ)` |
| Schur index | the (twisted) trace `Tr` |

## The five blocks

**A — the intrinsic (topological) layer.** Simple multicurves in normal
coordinates on an ideal triangulation (`multicurve`, `triangulation`,
`bordered_triangulation`), the classical Kauffman-bracket product by
stack–resolve–tighten (`skein_resolve`, `skein_algebra`), the stated skein
algebras of the bigon / triangle / disk / polygon (`stated_bigon`,
`stated_triangle`, `stated_disk`, `stated_polygon`), and the quantum-trace /
geometric-F dictionaries (`quantum_trace`, `geometric_f`, `kappa_qt`, `phi_map`,
`y_delta`). This block is **deliberately independent of the `KAlgebra`
contract**, so the skein ↔ BPS isomorphisms below are non-circular evidence, not
bookkeeping.

**B — the unified class.** `SkeinKAlgebra(ConeKAlgebra, BPSKAlgebra)`
(`skein_kalgebra`): the rays are simple non-self-intersecting curves (closed
curves → SU(2)-character generators, open arcs → monomial generators, adjoined
invertibles → torus pairs), the cross-products come from a certified skein
engine, and an *optional* BPS chart substrate (given-F / recursive-S) supplies
the realisation routes. A chart-less instance **fails honestly** on every BPS
route with a named diagnostic rather than guessing. Per-instance honesty flags
(`engine_provenance`, `trace_provenance`) record whether a product is *produced
by* a native skein computation or *verified against* a stated one.

**C — the roster.** Sixteen named per-theory instances: the `[A₁,Aₙ]` polygon
family (`U1Square` / `Pentagon` / `U1Hexagon` / `Heptagon` / …, and any `n ≥ 5`
via `SkeinKAlgebra.polygon(n)`), the pure SU(2) annulus, the fully native
four-punctured sphere = SU(2) `N_f=4` (native Kauffman engine + intrinsic
Schur–Askey–Wilson trace), the complete SU(2)+`N_f` family
`SkeinKAlgebra.su2_nf(0..4)` over bordered flip-triangulated charts, the
two-puncture disks, and the flavoured D-family (`a1d3`, `u1a1d4`).

**D — isomorphisms and objects.** Per-instance `build_iso()` witnesses
(skein-cone ↔ BPS twin); the skein legs on the shared `KAlgebraObject`s
(`sqed1_object`, `u1hexagon_object` — a `'skein-cone'` realisation beside the
Step-2/4/5 legs); and the **independent vacuum anchors**: the SU(2)+`N_f` vacuum
Schur indices computed through the skein charts match the strong-coupling BPS
quivers (`bps_su2_nf1/2/3` from Step 5), refined under measured integral flavour
frame maps. The four-punctured sphere measures `Tr 1 = 1 + χ_{28} 𝖖² + …`, the
Spin(8) branching.

**E — `SkeinAtlas`.** An ensemble of `SkeinKAlgebra` charts of a single `Sk(Σ)`
indexed by ideal triangulations, with transitions = diagonal flips ≡ FST
mutations, each certified per edge as a `KAlgebraIso`. It covers closed surfaces
(the tetrahedron and bipyramid charts of `S²` with four punctures), bordered
polygons (all odd `n`; the five-flip associahedron loop composes to `ρ²` at the
lifted level), gauged even polygons (with a construction-time meson gate), and
the flavoured D-family (the arc ↔ charge frames come from the Step-2
`finite_a1d{3,5,7}` ground-truth lattices). The closed tetrahedron chart
**measures** the three-term GNO tower `L_γ² = L_{2γ} + L_γ + 1`.

## What's included (`src/skein/`)

The intrinsic engine (`skein_algebra`, `skein_resolve`, `skein_product`,
`multicurve`, `multi_arc`, `multicurve_trace`, `triangulation`,
`bordered_triangulation`, `triangle_algebra`, `curve_realization`,
`half_laurent`), the stated-skein algebras (`stated_bigon`, `stated_triangle`,
`stated_disk`, `stated_polygon`, `pinned_polygon`, `pinned_pentagon_kalg`,
`pinned_closed`, `pinned_seam`), and the quantum-trace dictionaries
(`quantum_trace`, `geometric_f`, `kappa_qt`, `phi_map`, `y_delta`,
`schur_measure`).

The unified class and its roster (`skein_kalgebra`, `bordered_skein_kalg`,
`cone_rgkalgebra`, and the per-theory instances: the polygon family
`skein_pentagon`/`skein_pentagon_kalg`/`skein_square_kalg`/`skein_heptagon_kalg`/
`skein_oddgon_kalg`/`skein_evengon_kalg`/`skein_nonagon_kalg`/
`u1_hexagon_pentagon_kalg`/`skein_u1hexagon_kalg`/`u1a1aodd_general`, the sphere
and annulus theories `skein_sphere_kalg`/`skein_annulus_kalg`/
`su2_nf4_sample_kalgebra` and its character / trace / symmetry machinery
`su2_nf4_skein_chars`/`su2_nf4_trace`/`su2_nf4_symmetry`/`su2_nf4_so8`/
`su2_nf4_bracelet`/`spin8_characters`, the SU(2)+`N_f` disks and annuli
`skein_su2_nf1_annulus`/`skein_su2_nf2_disk`/`skein_su2_nf3_disk`/
`skein_su2_2punct_disk`/`su2_gauging`/`su2n_flavour`/`su2_nf4_symmetry`, and the
D-family `skein_a1d3_kalg`/`skein_a1dn_disk`/`skein_su2a1dn_annulus`/
`a1dodd_fork_quiver`).

The atlases, isomorphisms and objects (`skein_atlas`, `skein_sphere_atlas`,
`cut_reglue`, `atlas_examples`; the witnesses `skein_bps_iso`, `a1a2k_bps_iso`,
`u1_hexagon_bps_iso`, `hexagon_bps_iso_v2`; the object legs `sqed1_object`,
`hexagon_objects`; the cone data `skein_cone_data`).

## Tests

```bash
python3 run_tests.py        # the full gate; test_skein_flows.py runs last
```

`tests/test_skein_flows.py` exercises the tier in eight sections: **A** the
intrinsic Kauffman algebra (contract-free) on the four-punctured sphere; **B**
the two-parent architecture (a chart-ful instance vs a chart-less honest-fail);
**C** the named roster and the `polygon` / `su2_nf` dispatchers; **D** the
per-engine certification spots (pinned / native Kauffman with exact coefficients
/ bordered chart / flavoured Clebsch–Gordan); **E** the `KAlgebraIso` witnesses
and `KAlgebraObject` legs; **F** the contract axioms (bar, `ρ²`-twisted trace,
orthonormality) through the universal surface; **G** the `SkeinAtlas` charts
(the closed GNO tower, a bordered flip, a gauged and a flavoured chart); and
**H** the independent vacuum anchor — the skein-chart Schur index against the
strong-coupling BPS quiver, sharing no chart machinery.

## License

GPL-3.0-or-later (see `LICENSE`).
