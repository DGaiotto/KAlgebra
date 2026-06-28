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
algebra of SQED₂), a range of K-theoretic Coulomb-branch algebras, a live RG-flow
engine (`RGKAlgebra`) that computes a theory's algebra from its flow to a graded
auxiliary, and a BPS-quiver realisation engine (`BPSKAlgebra`) that builds a
theory's algebra directly from its BPS quiver.

## Organisation

A single package layered over one contract:

| directory | contents |
|---|---|
| `src/core/` | the `KAlgebra` contract (the primitives and the derived axiom verifiers), the Z₊-ring coefficient layer (for flavoured algebras), exact `Z[𝖖^±]` arithmetic, and the `KAlgebraIso` isomorphism witness |
| `src/samples/` | algebras implemented directly from the contract: the quantum torus, the pentagon `K_𝖖([A_1,A_2])`, and SQED₁/₂/_{N_f}_ |
| `src/cone/` | the `ConeKAlgebra` helper — a `KAlgebra` subclass that reduces the canonical basis to normal-ordered expressions in a set of multiplicative *ray* generators — together with the catalogue of realisations it presents |
| `src/rg/` | the `RGKAlgebra` engine — a `KAlgebra` whose entire API (`RG`, `multiply`, `ρ`, `trace`) is computed live from an RG flow to a graded auxiliary — and the catalogue of flows it presents (rank-1 Argyres–Douglas chains, Lagrangian SU(2) gauge theories, nested and formal flows) |
| `src/bps/` | the `BPSKAlgebra` engine — a `KAlgebra` realised from a BPS quiver (the Kontsevich–Soibelman spectrum generator + the `F·S = X_γ + O(𝖖)` discovery relation), with the cluster-mutation `BPSAtlas`. This is the realisation **spine**; Steps 1–3 are spine-free and never import it |
| `src/iso/` | `KAlgebraIso` witnesses identifying a sample algebra with its cone realisation |

Per-layer documentation is in `docs/`: `docs/step1-KAlgebra.md` (the contract and
the samples), `docs/step2-ConeKAlgebra.md` (the cone realisations),
`docs/step3-RGKAlgebra.md` (the live RG-flow engine), `docs/step4-BPSKAlgebra.md`
(the BPS-quiver realisation engine), `docs/conjectures-*.md` (the orthonormality
conjecture), and `docs/axioms-and-bootstrap.md` (how the axioms determine the
traces).

## Tests

```
python3 run_tests.py        # or: pytest
```

runs `tests/test_samples.py`, `tests/test_cones.py`,
`tests/test_sample_cone_iso.py`, the eight Step-3 RG self-tests
(`tests/test_rg_flows.py`, `test_a1an_chain.py`, `test_dn_chain.py`,
`test_e_type.py`, `test_flavoured_fork.py`, `test_over_pure.py`,
`test_su2_gauged_chain.py`, `test_wild.py`), and the Step-4 BPS self-test
(`tests/test_bps_flows.py`). These exercise the contract verifiers — the bar
involution, the unit law, associativity, the `ρ`-automorphism property,
`ρ²`-twisted trace cyclicity, and orthonormality — on the sample algebras, the
cone realisations, the sample-to-cone isomorphisms, the live RG flows, and the BPS
realisation. The Step-3 RG self-tests additionally assert that no realisation-spine
module is imported, so a green run certifies those layers are spine-free;
`test_bps_flows.py` is run last, because Step 4 *is* the spine.

The modules import one another by unqualified name; `conftest.py` and
`run_tests.py` place each `src/` subdirectory on `sys.path`. The modules must
therefore not be nested further, nor given `__init__.py` files.

## Arithmetic

Structure constants, traces, and pairings are computed in exact `Z[𝖖^±]` and
`R[𝖖^±]` arithmetic. Truncation to `O(𝖖^K)` occurs only where a trace is read
out.

## License

GPL-3.0-or-later; see `LICENSE`.
