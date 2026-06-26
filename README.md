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
algebra of SQED₂), and a range of K-theoretic Coulomb-branch algebras.

## Organisation

A single package layered over one contract:

| directory | contents |
|---|---|
| `src/core/` | the `KAlgebra` contract (the primitives and the derived axiom verifiers), the Z₊-ring coefficient layer (for flavoured algebras), exact `Z[𝖖^±]` arithmetic, and the `KAlgebraIso` isomorphism witness |
| `src/samples/` | algebras implemented directly from the contract: the quantum torus, the pentagon `K_𝖖([A_1,A_2])`, and SQED₁/₂/_{N_f}_ |
| `src/cone/` | the `ConeKAlgebra` helper — a `KAlgebra` subclass that reduces the canonical basis to normal-ordered expressions in a set of multiplicative *ray* generators — together with the catalogue of realisations it presents |
| `src/iso/` | `KAlgebraIso` witnesses identifying a sample algebra with its cone realisation |

Per-layer documentation is in `docs/`: `docs/step1-KAlgebra.md` (the contract and
the samples), `docs/step2-ConeKAlgebra.md` (the cone realisations),
`docs/conjectures-*.md` (the orthonormality conjecture), and
`docs/axioms-and-bootstrap.md` (how the axioms determine the traces).

## Tests

```
python3 run_tests.py        # or: pytest
```

runs `tests/test_samples.py`, `tests/test_cones.py`, and
`tests/test_sample_cone_iso.py`. These exercise the contract verifiers — the bar
involution, the unit law, associativity, the `ρ`-automorphism property,
`ρ²`-twisted trace cyclicity, and orthonormality — on the sample algebras, the
cone realisations, and the sample-to-cone isomorphisms.

The modules import one another by unqualified name; `conftest.py` and
`run_tests.py` place each `src/` subdirectory on `sys.path`. The modules must
therefore not be nested further, nor given `__init__.py` files.

## Arithmetic

Structure constants, traces, and pairings are computed in exact `Z[𝖖^±]` and
`R[𝖖^±]` arithmetic. Truncation to `O(𝖖^K)` occurs only where a trace is read
out.

## License

GPL-3.0-or-later; see `LICENSE`.
