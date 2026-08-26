# K_𝖖-algebras

A dependency-free (pure Python 3) implementation of **K_𝖖-algebras** and a range
of worked examples.

A K_𝖖-algebra axiomatises the fusion algebra of rotation-equivariant BPS line
defects in a 4d 𝒩=2 theory. It is an associative algebra `A_𝖖` over `Z[𝖖^±]`,
free as a `Z[𝖖^±]`-module, equipped with

- a **bar involution** — an antimultiplicative involution sending `𝖖 ↦ 𝖖⁻¹`;
- a **canonical basis** `{L_a}` of bar-invariant elements, containing the unit;
- an algebra **automorphism `ρ`** permuting the canonical basis; and
- a **`ρ²`-twisted trace** `Tr`, under which the pairing
  ```
  I_{a,b} = Tr(L_{ρ(a)} · L_b) = δ_{a,b} + O(𝖖)
  ```

The final relation — orthonormality of the canonical basis to leading order in
`𝖖` — is the defining constraint, and it is rigid enough to determine the trace
(`docs/axioms-and-bootstrap.md` explains how). The abstract contract is the
`KAlgebra` class; the examples implemented here include the quantum torus
`Q_𝖖(Γ)`, the pentagon `K_𝖖([A_1,A_2])`, the SU(2)-flavoured `U_𝖖(𝔰𝔩₂)` (the
algebra of SQED₂), the abelian gauge theories SQED₁/₂/_{N_f}_, the U(1)-gauged
Argyres–Douglas families, and SU(2) gauge-theory cone algebras (the K-theoretic
Coulomb branch algebras of those conventional gauge theories), a live RG-flow
engine (`RGKAlgebra`) that computes a theory's algebra from its flow to a graded
auxiliary, a BPS-quiver realisation engine (`BPSKAlgebra`) that builds a
theory's algebra directly from its BPS quiver, and an abelianized-presentation
tier (`AbeKAlgebra`) that presents the K-theoretic Coulomb branch algebras of
conventional gauge theories — pure U(N)/SU(N), U(N)+N_f matter, quivers — on an
enriched rational quantum torus, with an object layer certifying each algebra `≅`
its cone / BPS / RG presentation, a skein tier (`SkeinKAlgebra`) that
realises the SU(2) skein algebras of marked surfaces as those same algebras,
with a flip-atlas whose diagonal flips are certified quiver mutations, and a
general-gauge-group tier (`PureGAbeKAlgebra` / `GNAbeKAlgebra`) that carries the
abelianized presentation to gauge theory at an arbitrary 4d gauge group with
arbitrary matter.

## Organisation

A single package layered over one contract:

| directory | contents |
|---|---|
| `src/core/` | the `KAlgebra` contract (the primitives and the derived axiom verifiers), the Z₊-ring coefficient layer (for flavoured algebras), exact `Z[𝖖^±]` arithmetic, and the `KAlgebraIso` isomorphism witness |
| `src/samples/` | algebras implemented directly from the contract: the quantum torus, the pentagon `K_𝖖([A_1,A_2])`, and SQED₁/₂/_{N_f}_ |
| `src/cone/` | the `ConeKAlgebra` helper — a `KAlgebra` subclass that reduces the canonical basis to normal-ordered expressions in a set of multiplicative *ray* generators — together with the catalogue of realisations it presents |
| `src/rg/` | the `RGKAlgebra` engine — a `KAlgebra` whose entire API (`RG`, `multiply`, `ρ`, `trace`) is computed live from an RG flow to a graded auxiliary — and the catalogue of flows it presents (rank-1 Argyres–Douglas chains, Lagrangian SU(2) gauge theories, nested and formal flows) |
| `src/bps/` | the `BPSKAlgebra` engine — a `KAlgebra` realised from a BPS quiver (the Kontsevich–Soibelman spectrum generator + the `F·S = X_γ + O(𝖖)` discovery relation), with the cluster-mutation `BPSAtlas`, the builder that produces `S` from the quiver alone as one BPS factor `E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}` per `(γ, s)` pair, a search over the order those factors are placed in, and the joint builder that grows `F_γ` and `S` together out of the discovery relation. This is the realisation **spine**; Steps 1–3 are spine-free and never import it |
| `src/abe/` | the `AbeKAlgebra` tier — a `KAlgebra` presented faithfully on an enriched rational quantum torus: the abelianized (gauge-fugacity) description of the K-theoretic Coulomb branch algebras of conventional gauge theories (pure U(N)/SU(N), U(N)+N_f matter, linear quivers), the N=2\* canonical finder, and the `KAlgebraObject` capstone that certifies each algebra `≅` its cone / BPS / RG presentation. Relies on Steps 1–4 by design |
| `src/skein/` | the `SkeinKAlgebra` tier — the SU(2) (Kauffman-bracket) skein algebras of marked surfaces realised as `A_𝖖[T[A₁, Σ]]`: an intrinsic (contract-free) topological engine, the two-parent `SkeinKAlgebra(ConeKAlgebra, BPSKAlgebra)` class, a roster of named theories (the `[A₁,Aₙ]` polygons, the four-punctured sphere = SU(2) N_f=4, the SU(2)+N_f family, the D-family), and the `SkeinAtlas` whose diagonal flips are certified quiver mutations. Relies on Steps 1–5 by design |
| `src/gn/` | the general-gauge-group tier — gauge theory at an arbitrary 4d gauge group with arbitrary matter: `PureGAbeKAlgebra` (pure gauge over any `RootDatum`, where the axioms alone — the leading-orbit seed, bar, `O(𝖖)` and the (★)-guarded solve — build **every** label, Wilson included, which is what opens the groups with no minuscule cocharacter), `GNAbeKAlgebra` (`T^*N` matter at any datum and any matter *representation*, with iterable matter-removal flows), `LineLattice` (the 4d gauge group data as a lattice and its dual — including the *correlated* forms, so `su(2)` has three — with the Langlands family `(G, Adj) ↔ (G^∨, Adj)` as certified `KAlgebraIso`), `line_lattice_torus` (that lattice's own conventional and rational quantum tori), `AuxSpace` (the Schur pairing in vacuum-state form, with `ρ` never constructed), and `PureSO3KAlgebra` (the independent BPS-quiver oracle that checks the odd-height claim from outside the tier — the one module here that imports the Step-4 spine). Relies on Steps 1–6 by design |
| `src/iso/` | `KAlgebraIso` witnesses identifying a sample algebra with its cone realisation |

Per-layer documentation is in `docs/`: `docs/step1-KAlgebra.md` (the contract and
the samples), `docs/step2-ConeKAlgebra.md` (the cone realisations),
`docs/step3-RGKAlgebra.md` (the live RG-flow engine), `docs/step4-BPSKAlgebra.md`
(the BPS-quiver realisation engine), `docs/step5-AbeKAlgebra.md` (the abelianized
presentation tier and the object-layer capstone), `docs/step6-SkeinKAlgebra.md`
(the skein-algebra tier and its flip-atlas), `docs/step7-GNKAlgebra.md` (the
general-gauge-group tier and the (★)-guarded build), `docs/conjectures-*.md` (the
orthonormality, one-object-many-presentations, and skein-dictionary conjectures),
and `docs/axioms-and-bootstrap.md` (how the axioms determine the traces).

Two notes bound what the gate establishes:
`docs/verification-scope.md` (what a green gate does *not* certify — the two
best-effort RG checkers it omits, and the one search bound that is proven only
in the pointed case) and `docs/frozen-data-provenance.md` (where the eight
`.pkl` tables and the fourteen generated `finite_*_kalg.py` modules came from,
their content hashes, and what validates them here given that their builders
are not shipped).

## Tests

```
python3 run_tests.py
```

is the validation gate. (`pytest` also runs, but does not cover the full gate:
it skips `test_cones.py` and `test_sample_cone_iso.py`, and importing the BPS
suite at collection time defeats the spine-freeness assertions of the Step-3
suites — use `python3 run_tests.py` to certify everything.) The gate
runs `tests/test_samples.py`, `tests/test_cones.py`,
`tests/test_sample_cone_iso.py`, the eight Step-3 RG self-tests
(`tests/test_rg_flows.py`, `test_a1an_chain.py`, `test_dn_chain.py`,
`test_e_type.py`, `test_flavoured_fork.py`, `test_over_pure.py`,
`test_su2_gauged_chain.py`, `test_wild.py`), the Step-4 BPS self-test
(`tests/test_bps_flows.py`), the Step-5 abelianized-tier self-test
(`tests/test_abe_flows.py`), the Step-6 skein-tier self-test
(`tests/test_skein_flows.py`), and the Step-7 general-gauge-group self-test
(`tests/test_gn_flows.py`). These exercise the contract verifiers — the bar
involution, the unit law, associativity, the `ρ`-automorphism property,
`ρ²`-twisted trace cyclicity, and orthonormality — on the sample algebras, the
cone realisations, the sample-to-cone isomorphisms, the live RG flows, the BPS
realisation, the abelianized presentations, the skein algebras, and the
general-gauge-group charts. Seven of the
eight Step-3 RG self-tests (all but `test_rg_flows.py`) additionally assert that
no realisation-spine module is imported; `test_bps_flows.py`, `test_abe_flows.py`,
`test_skein_flows.py` and `test_gn_flows.py` are run last, because Steps 4–6
import the spine and Step 7 reaches it through the one module that must — the
`PureSO3KAlgebra` oracle it checks its odd-height claim against.

The modules import one another by unqualified name; `conftest.py` and
`run_tests.py` place each `src/` subdirectory on `sys.path`. The modules must
therefore not be nested further, nor given `__init__.py` files.

## Arithmetic

Structure constants, traces, and pairings are computed in exact `Z[𝖖^±]` and
`R[𝖖^±]` arithmetic. Truncation to `O(𝖖^K)` occurs only where a trace is read
out.

## License

GPL-3.0-or-later; see `LICENSE`.
