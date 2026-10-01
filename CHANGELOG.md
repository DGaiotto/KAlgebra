# Changelog

## The release accompanying version 1 of the paper

Changes relative to the previous public version (2026-08-17).  The repository
now supports the paper: its companion (`paper_companion.tex`) describes the
implementation and testing of every claim, and the battery re-runs them.

### Added

- **`paper/`** — the paper's LaTeX source, as submitted to the arXiv:
  `nice_temp.tex`, its bibliography `nice_temp.bbl`, `preamble_ams.tex` and
  `JHEP.bst`.  The companion takes the paper's theorem and equation numbers
  from it.
- **`battery/`** — the claim registry (`claims.json`, one entry per claim of the
  paper, keyed by its labels), the runner, the renderer of the companion's
  appendix, the adapters (`checks_*.py`), and the result records the appendix
  was rendered from.  `experiments/` holds the five research probes the
  adapters import.  See `battery/README.md`.
- **`dictionaries/`** — the BPS-quiver dictionaries, with their builders and
  the loader `dictionary_loader`: the enumerated dictionary (every strongly
  connected quiver of total arrow weight at most 12, each with one accepted
  spectrum-generator sequence or its crystalline certification; every other
  quiver is answered by composing its strongly connected components) and the
  flavoured family (ten tiers, by a manifest flavour symmetry from `SU(2)` to
  `SU(2)³`).  The seed-closure dictionary is built on demand
  (`dictionaries/build.py`) and not shipped.
- **`notes/`** — the design notes the paper's derived claims cite, verbatim,
  under a banner: their internal references point into the source repository.
- **A second gate tier**: `python3 run_tests.py --cited` also runs the 96 tests
  the paper companion cites, each in its own process (`tests/cited.txt`).
- **Library modules the companion and its battery reach**, among them:
  `UqSU2KAlgebra` with its certified isomorphism to the `SU(2)`-flavoured SQED₂
  chart (`uq_su2_kalgebra`, `uq_su2_bps_iso`), the finite zoo's object layer
  (`finite_kalgebra_objects` — the package module `finite_kalgebras/objects` of
  the source repository, under a flat name of its own), the closed-form seed
  modules of the finite-type families (`aeven_seeds`, `aodd_seeds`,
  `a1d3_seeds`, `a1d4_seeds`, `a1dodd_seeds`, `a1deven_seeds`, `e7_seeds`,
  `w3_seeds`), the geometric labels of the A and D families (`zoo_geometry`),
  the flavoured spectrum machinery (`flavoured_factor_spectrum`,
  `flavoured_f_solver`, `flavoured_spec`), `spec_acceptance`,
  `quiver_enumeration`, `mutation_s_relation` (the relation between the
  spectrum generators of a quiver and of its mutation, for the paper's
  conjecture on mutations), and the atlases of the gauged and quiver families.

### Changed

- **The contract gains an axiom, ρ-equivariance of the trace**:
  `Tr ∘ ρ = ⋆ ∘ Tr`, where `⋆` is the duality of the flavour representation
  ring acting on the coefficients (not bar: `𝖖` is untouched).  Its pairing
  form is `I_{b,a} = ⋆(I_{a,b})`.  Verifiers:
  `KAlgebra.verify_trace_intertwines_rho`, `verify_trace_intertwines_rho_star`,
  `verify_pairing_rho_star_symmetric`, and `RGKAlgebra.verify_rg_inherits_rho_star`
  for its inheritance along an RG flow; `RPowerSeries.star` and
  `RingHom.verify_commutes_with_star` on the coefficient layer.
- **General gauge group: one production route.**  `PureGAbeKAlgebra` and
  `GNAbeKAlgebra` build every canonical element by the (★)-guarded solve —
  Wilson lines included, as the case with a single support cell.  The earlier
  constructive routes are kept behind `constructive_routes=True` as the
  independent construction to compare against, and `closed_form` is excluded
  from production (at `SU(3)`, `m = (2,2)` it returns a wrong element without
  raising).  Named optimizations are declared through `optimizations=`: by
  default `theta_twist`, and `monoid` in pure gauge theory, each asserted to
  return the same element as the axiom route; `spectral` (the undressed
  elements from the spectral law, `spectral_transcription`) on request;
  `optimizations=()` is the bare axiom route.  `route(label)` reports which route built a label.
- **Persistent charts on the abelianized tier**: `AbeKAlgebra.save_cache` /
  `load_cache`, with a presentation fingerprint and every loaded element
  re-verified.
- **ρ on labels in closed form** (`wrq_torus.rho_label`); the read-back through
  the torus twist becomes the verifier `AbeKAlgebra.verify_rho_via_twist`.
  `GNAbeKAlgebra` exposes `rho` / `rho_inverse` directly.
- **The sector measure in closed form** (`wrq_torus.sector_measure_closed_form`,
  with its verifiers), derived for every `G` and `(G, N)`.
- **Global forms**: `LineLattice` carries the centre classes of its lines
  (`label_classes`, `class_pairs`, `is_maximal`, `verify_dirac_integral`, and a
  `classes=` argument), which admits the intermediate forms such as
  `SU(4)/Z₂`.
- **BPS spectrum generators on quivers that are not strongly connected**: the
  spectrum-generator search (`BPSQuiver.find_negating_sequence`,
  `find_spec_auto`, argument `decompose=`) splits the quiver into its strongly
  connected components and searches each alone; the factor engine's default
  order on such a quiver is the component order.
- **The finite-type algebras carry no frozen trace data.**  Every `[A_1,ADE]`
  finite-type algebra, and its `U(1)`-gauged version where the flavour has a
  `U(1)`, is realised from its own data: products and traces on every label, to
  any order.  The A and D families carry geometric canonical-basis labels
  (`geometric_label`: curves, or balanced multisets of curves, on the polygon
  of the theory).  `U1A1AoddKAlg(k)` is analytic at every `k`;
  `U1A1DevenConeKAlgebra` is the curve frame, labels `(curves, e, κ)`.  The only
  stored tables left are `U1E7ConeKAlgebra`'s product and ρ tables.
- **Objects**: the pure `SU(2)` object gains a `skein` presentation; the
  `SU(2)+N_f` object has the presentations `{cone, bps}` (its abelianized one
  went with `UNNfKAlgebra`, below; restoring it on `GNAbeKAlgebra` is open).

### Removed

The type-A abelianized classes, superseded by the general gauge-group tier:

| module | replacement |
|---|---|
| `pure_un_kalgebra` | `PureGAbeKAlgebra(u_n(N))`, measured equal at `N = 2, 3` |
| `pure_sun_kalgebra` | `PureGAbeKAlgebra(su_n(N))` (the retired class returned wrong products at dressed labels) |
| `un_nf_kalgebra` | `GNAbeKAlgebra(u_n(N), fundamental, nf=N_f)` and the presets of `g_matter_roster` |
| `un_quiver_kalgebra` | `GNAbeKAlgebra` on a product datum with bifundamentals |
| `un_nf_over_pure_rgflow`, `quiver_over_pure` | `GMatterOverPure` / `GMatterOverMatter` (matter removal at any `G`) |
| `su2_nf1_abe`, `su2_unf_abe_kalgebra` | `SU2Nf1FlavourInCoefficients` (`su2_nf1_object`); the presets `su2-nf1` / `su2-nf2` of `g_matter_roster` |
| `pure_via_n2star` | the (★)-guarded solve (`star_bubbling`) |
| `g_matter_un_nf_seam` | nothing: the general tier is the only `U(N)+N_f` presentation; the flow isomorphism is certified where the irreps are distinct |
| `pure_sun_bps_iso`, `pure_un_canonical`, `pure_un_chart_engine`, `pure_un_closed_form`, `pure_un_construct`, `sun_rq_torus`, `un_nf1_over_pure_iso`, `un_nf_bps_iso`, `un_nf_dressed_generators` | the general tier (`PureGAbeKAlgebra` / `GNAbeKAlgebra`); these existed for the retired classes |

The stand-alone and frozen finite-type modules, superseded by the classes
above that need no stored data:

| module | replacement |
|---|---|
| `u1_octagon_*`, `u1_decagon_*`, `u1_dodecagon_*` (kalg, cone data, multiplication table; the decagon's singlet), `u1a1aodd_k4_chord_charges` | `U1A1AoddKAlg(k)`, `k = 2, 3, 4` |
| `u1a1deven_cone_build`, `u1a1deven_dodd_build`, `u1a1deven_matter_bootstrap` | `U1A1DevenConeKAlgebra` (closed forms) |
| `a1dodd_cone_tables` | the closed-form arc rules of `a1dodd_cone_data` |
| `u1a1aodd_general` (skein tier) | the gauged chart from `u1a1aodd_kalg` |
| the tables `u1a1aodd_tables_k{1,2,3,4}.pkl`, `u1a1deven_tables_k1.pkl`, `u1a1deven_traces_k1.pkl`, and the ten frozen trace tables of `elem_trace_data` | the realisations' own traces |
