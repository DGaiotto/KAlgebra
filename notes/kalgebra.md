# KAlgebra — the `A_𝖖[T]` contract + BPS-quiver realization

This is the design doc for the **canonical repo-root head** of the
codebase: the abstract notion of a **K-theoretic Coulomb branch
algebra** — a *`K_𝖖`-algebra* in the sense of Ambrosino–Gaiotto
(DESY-25-035, `main.tex` §"K_𝖖-algebras") — written `A_𝖖[T]`, and its
concrete realization through a BPS-quiver chart.  Since Plan 07
(2026-05-10) these modules live at the **repo root** (`kalgebra.py`,
`bps_kalgebra.py`, `zplus_ring.py`, …); the preliminary stack they
replaced is archived under `legacy/`.  The `KAlgebra` ABC is the
logical head of the repo — the contract every concrete `A_𝖖[T]`
specialises.

## Files

### Abstraction

| File | Contents |
|---|---|
| `zplus_ring.py` | `ZPlusRing` ABC (Lusztig-Ostrik Z₊-rings) + concrete coefficient rings: `TrivialZPlusRing` (= `Z`, unflavoured), the abelian `AbelianZPlusRing(rank=n)` (= `R(U(1)ⁿ) = Z[μ_1^±, …, μ_n^±]`), and the non-abelian rep rings `SU2ZPlusRing`, `SO3ZPlusRing`, `SU3ZPlusRing` (Clebsch–Gordan / Klimyk–Freudenthal multiplication).  `RElement` for ring elements, `RLaurent` for `R[q^±]`, `RPowerSeries` for truncated `R((q))`.  `RingHom` + factories `identity_hom` / `augmentation_hom` / `restriction_hom` / `so3_to_u1_hom` / `so3_to_su2_hom` for `base_change`.  Bar (KAlgebra-level) is q-flip only on `RLaurent`; ⋆ (Z₊-ring-level, = rep-ring duality / ρ-on-central) acts on the R-side via `RElement.star()` and `RLaurent.star()`.  See "Flavour in `KAlgebra`" below. |
| `kalgebra.py` | **Unified parameterized** `KAlgebra` ABC over a Z₊-ring `R`.  Per Plan 10, the canonical basis is over `Z[q, q⁻¹]` (universal); `Element` stores `dict[Label, LaurentPoly]` (no ring field).  The R-module structure is an *API view*: `to_R_form(x: Element) -> ElementOverR` produces `dict[Label, RLaurent]` over a chosen R via the `_label_section_decompose` bridge, which factors a label as `(section_rep, flav_coords)` so flavour shifts collapse onto μ-monomial R-coefficients.  Six abstract primitives: `coefficient_ring`, `identity`, `multiply`, `rho`, `rho_inverse`, `trace`, plus the optional flavour-lift coordinate `r_label_decompose` (the single-irrep `(section, R-basis-label)` hook; since the 2026-06-19 migration #555 this is *the* flavour-lift primitive and `_label_section_decompose` is a derived legacy bridge `to_R_form` still reads).  ρ is a strict basis-permuting automorphism (no twist at the Z-form level).  Bar (q-flip) lives on `LaurentPoly`; ⋆ (R-side, rep-ring duality) lives on `RElement` / `RLaurent` and only surfaces inside `to_R_form` / R-side trace boundary.  No `quantum_torus` or `schur_index` imports. |
| `kalgebra_samples.py` | Four hard-coded sample K-algebras: `TrivialKAlg`, `QuantumTorusZ2KAlg`, `Sqed1KAlg`, `PentagonKAlg`.  All four sit at `R = TrivialZPlusRing` (unflavoured).  HabiroElement / qpoch_infty / pentagon_algebra still imported from repo (q-series tools; eliminating those is a future cleanup), with boundary conversion to `RLaurent` / `RPowerSeries` at the contract surface. |

### BPS-quiver realization

| File | Contents |
|---|---|
| `bps_kalgebra.py` | `BPSKAlgebra(KAlgebra)`: trimmed surface (KAlgebra contract + `F`, `spectrum_generator`, `shorten_spec()`).  Owns a private lazy chart graph. |
| `bps_kalgebra_internals.py` | Solve-F, Schur-index, F·S-coefficient helpers parameterized by an `s_coefficient` callable.  Includes the (b)-shortcut machinery (`solve_F_modified`, `s1_xdelta_s2_coeff`) called from `_try_solve_F_via_b` inside (c). |
| `chart_graph.py` | `Chart`, `Mutation`, `ChartGraph` -- the lazy graph used by `BPSKAlgebra._solve_F_via_chain` (algorithm c) and `_multiply_via_chart_search`.  Tracks `(s1_chain, boundary_preserved, s2_length)` per chart so the (b)-shortcut can fire at chart-end. |
| `spec_shortening.py` | Local-move shortening (pentagon collapse + commute swap + pentagon expand) + cone helpers. |
| `bpskalgebra_iso.py` | Strong + weak isomorphism witnesses between two `BPSKAlgebra`s (cross-lattice supported: `A^T B_2 A = B_1`). |

### Tests

| File | Tests |
|---|---|
| `tests/test_zplus_ring.py` | `ZPlusRing` protocol + `TrivialZPlusRing` + `AbelianZPlusRing` + `RElement` + `RLaurent` + `RPowerSeries` (80). |
| `tests/test_kalgebra.py` | Parameterized KAlgebra ABC + 4 unflavoured samples + `MuKAlg` (non-trivial-R sanity for the ρ-twist) (17). |
| `tests/test_bps_kalgebra.py` | BPSKAlgebra (26, including chart-graph wiring + algorithm-c agreement + chart-search-multiply agreement). |
| `tests/test_chart_graph.py` | Chart-graph machinery (13). |
| `tests/test_bpskalgebra_iso.py` | Strong + weak iso witnesses + cross-lattice (23). |

Run all (from the repo root):
```
PYTHONPATH=. python tests/test_kalgebra.py
PYTHONPATH=. python tests/test_bps_kalgebra.py
PYTHONPATH=. python tests/test_chart_graph.py
PYTHONPATH=. python tests/test_bpskalgebra_iso.py
```

### Design notes

| File | Notes |
|---|---|
| `kalgebra.md` | This file. |
| `bpskalgebra_chart_graph.md` | Chart-graph design + algorithm sketches. |

#### ρ is piecewise-linear, NOT Z-linear, on infinite-T-tower algebras

**Recurring confusion alert** — every time you build a U(1)-flavoured
(or unflavoured) BPS K-algebra whose half-monodromy ρ has infinite
order on the gauge T-tower, you'll wonder "why isn't ρ just a Z-linear
matrix on Γ?".  Answer: **it can't be**.

The empirical pattern (e.g. the SQED₂ BPS chart — `BPSKAlgebra` with
`pairing=[[0, 1, 0], [-1, 0, 0], [0, 0, 0]]` and
`node_charges=[(0, 1, 1), (0, 1, -1)]`, whose orbit is exactly the one
printed below (measured 2026-09-23; first recorded on the hand-written
`U1A1D2KAlg`, now retired to `legacy/u1_a1d2_kalg.py`) — `pSU2KAlgebra`,
`U1HexagonKAlg`): the ρ-orbit of T_0 = `(1, 0, 0, …)` looks like

    T_0 = (1, 0),  T_1 = (-1, -2),  T_2 = (1, 2),  T_3 = (-1, -4),
    T_4 = (1, 4),  T_5 = (-1, -6),  T_6 = (1, 6),  …

with step deltas alternating `(-2, -2), (+2, +4), (-2, -6), …` — the
γ_2-increment grows linearly in n, so no single Z-linear matrix
realises the entire ρ.  The structure IS **piecewise-linear**: linear
on each Weyl chamber (one chamber per ρ-orbit step), with chamber-
boundary transitions encoded by which T_n letter the label belongs to.

The fix in every case: implement ρ via **letter decomposition** rather
than a lattice formula:
  1. Decompose label into primitive letters (T_n^a · D^d · μ^p).
  2. Apply ρ letter-wise (ρ(T_n) = T_{n+1}, ρ(D_i) = D_{i+1}, ρ(μ) = μ^{-1}).
  3. Recompose to a lattice label via multiplication.

This is the U(1)-side analogue of the "(m, e)-coords" trick in
`pSU2KAlgebra` (= ρ shifts e by -4m, Z-linear in chamber-adapted
coords but non-linear in ambient Γ).

## The `K_𝖖`-algebra contract

Following Ambrosino–Gaiotto (DESY-25-035, §"K_𝖖-algebras"), a
**`K_𝖖`-algebra** is an associative algebra `A_𝖖` over `Z[𝖖, 𝖖⁻¹]`,
free as a `Z[𝖖, 𝖖⁻¹]`-module, endowed with:

1. **Bar involution** — an *antimultiplicative* `Z`-algebra involution
   sending `𝖖 ↦ 𝖖⁻¹`, i.e. `A_𝖖 ≅ A_{𝖖⁻¹}^op`.
2. **Canonical basis** `{L_a}` over `Z[𝖖, 𝖖⁻¹]` of *bar-invariant*
   elements, including the identity `1`.  Antimultiplicativity +
   bar-invariance of the basis is exactly the structure-constant
   constraint `C^c_{ab}(𝖖⁻¹) = C^c_{ba}(𝖖)`.
3. **Algebra automorphism** `ρ` permuting the basis labels and fixing
   `1`:  `ρ(L_a) = L_{ρ(a)}`.
4. **`ρ²`-twisted trace** `Tr : A_𝖖 → Z((𝖖))`, linear over
   `Z[𝖖, 𝖖⁻¹]`, with
   `I_{a,b} := Tr(L_{ρ(a)} L_b) = Tr(L_a L_{ρ⁻¹(b)}) = δ_{a,b} + O(𝖖)`.
   The two ways of writing `I_{a,b}` encode `ρ²`-twisted cyclicity
   `Tr(uv) = Tr(ρ²(v) u)` together with orthonormality; the second
   form `Tr(L_a L_{ρ⁻¹(b)})` is *why `ρ⁻¹` is a contract primitive*.
   The trace is only defined up to an overall rescaling by `1 + O(𝖖)`;
   physics fixes a canonical normalization (e.g. the `(𝖖²)_∞^{rk Γ}`
   prefactor on the quantum torus below).
5. **ρ-equivariance of the trace** (user, 2026-09-18): `Tr ∘ ρ = ⋆ ∘ Tr`, i.e.
   `Tr(L_{ρ(a)}) = ⋆(Tr(L_a))`, with `⋆` the Z₊-ring rep-ring duality on the
   coefficients (`RPowerSeries.star`; *not* bar — `𝖖` is untouched).  ρ acts on
   the target `R((𝖖))` through its own restriction to the centre, which is `⋆`.
   Equivalently, in pairing form, `I_{b,a} = ⋆(I_{a,b})` — at trivial flavour
   the Gram matrix is symmetric, with flavour it is Hermitian under
   `μ ↦ μ⁻¹`.  See "Axiom 5" below for the equivalence, the derivation, and
   where it is inherited versus where it is genuine content.

**Canonical example — the quantum torus** `Q_𝖖(Γ)` for a lattice `Γ`
with non-degenerate antisymmetric pairing: `X_γ X_{γ'} = 𝖖^{⟨γ,γ'⟩}
X_{γ+γ'}`, bar fixes each `X_γ`, `ρ_Q(γ) = −γ` (so `ρ² = id`), and
`Tr X_γ = (𝖖²)_∞^{rk Γ} δ_{γ,0}`, giving `I_{γ,γ'} = (𝖖²)_∞^{rk Γ}
δ_{γ,γ'}`.  Flavoured `K_𝖖`-algebras (next section) generalise the
coefficient `Z` to a representation ring `R_{G_f}`.

This is encoded faithfully in **six abstract primitives** plus an
**optional flavour-lift coordinate** (and derived helpers + axiom
verifiers).  (As of the 2026-06-19 migration #555 the flavour-lift
primitive is the single-irrep `r_label_decompose`; the older
`_label_section_decompose` is a **derived legacy bridge** — not
`@abstractmethod` — kept because `to_R_form` reads it and it is the
honest fallback for the deprecated emergent flavour diagnostics and the
non-contract lattice scaffolds.)

```python
class KAlgebra(ABC):
    @abstractmethod def coefficient_ring(self) -> ZPlusRing      # R = R_{G_f}; TrivialZPlusRing (= Z) if unflavoured
    @abstractmethod def identity(self) -> Label
    @abstractmethod def multiply(self, a, b) -> Element          # C^c_{ab}(𝖖) ∈ Z[𝖖^±]
    @abstractmethod def rho(self, a) -> Label
    @abstractmethod def rho_inverse(self, a) -> Label
    @abstractmethod def trace(self, a, K) -> RPowerSeries        # in R((𝖖)); R-content from the centre

    # optional 7th primitive: the flavour-lift coordinate (default raises;
    # trivial-flavour returns (label, R.one_basis()) universally)
    def r_label_decompose(self, label) -> tuple[Label, BasisElement]  # (section, single-irrep), L_label = χ_w·L_section
    def _label_section_decompose(self, label) -> tuple[Label, RElement]  # derived legacy bridge (since #555); to_R_form reads it

    # derived (default)
    def multiply_elements(x, y) -> Element
    def rho_element(x) -> Element                                # linear extension (no twist over Z)
    def rho_inverse_element(x) -> Element
    def trace_element(x, K) -> RPowerSeries
    def inner_product(a, b, K) -> RPowerSeries                   # trace pairing I_{a,b} = Tr(ρ(L_a)·L_b) = Tr(L_b·ρ⁻¹(L_a)); OVERRIDABLE (Schur-quantization pairing; default = multiply-then-trace)
    def to_R_form(x: Element) -> ElementOverR                    # R-module view via _label_section_decompose
    def base_change(phi: RingHom) -> KAlgebra                    # functorial flavour restriction R_1 → R_2
    def add_flavour(flavour) -> KAlgebra                         # dual: adjoin spectator flavour (R_self ⊗ R; any ZPlusRing) — flavoured_kalgebra.py

    # free-R-module API (the non-canonical free-R-module structure; see flavour section)
    def section_decompose(label) -> tuple[Label, RElement]       # public face of _label_section_decompose
    def embed_R(r: RElement) -> Element                          # central embedding R ↪ A (concrete, default 1_R ↦ identity; flavoured realisations override)
    def from_R_form(x: ElementOverR) -> Element                  # lift = Σ embed_R(c)·L_s (derived; not a primitive)
    def rho_R_form(x: ElementOverR) -> ElementOverR              # ρ on R-form, derived = to_R_form ∘ rho_element ∘ from_R_form

    # axiom verifiers (return bool)
    def verify_identity_in_basis()
    def verify_rho_fixes_identity()
    def verify_rho_inverse(a)
    def verify_rho_is_automorphism(a, b)        # ρ(L_a L_b) == ρ(L_a) ρ(L_b)
    def verify_bar_involution(a, b)             # multiply(a,b).bar() == multiply(b,a)
    def verify_rho_twisted_trace(a, b, K)       # Tr(ab) == Tr(ρ²(b) a)
    def verify_orthonormality(a, b, K)          # I_{a,b} == δ_{a,b} + O(𝖖)  (incl. empty negative window)
    def verify_trace_pairing_faces(a, b, K)     # Tr(ρ(a)·b) == Tr(b·ρ⁻¹(a))  (cyclicity, pairing form)
    def verify_inner_product_consistent(a, b, K)# inner_product override == multiply-then-trace default
    def verify_embed_section_roundtrip(a)       # embed_R(r_coeff(a))·L_{section(a)} == L_a
    def verify_embed_intertwines_rho(r)         # embed_R(r⋆) == ρ(embed_R(r))
    def verify_trace_intertwines_rho_star(a, K) # AXIOM 5: Tr(ρ(a)) == ⋆Tr(a)  (output-side twin of the line above)
    def verify_pairing_rho_star_symmetric(a, b, K)  # its pairing form  I_{b,a} == ⋆(I_{a,b})
```

**`_label_section_decompose` returns `(section_rep, r_coefficient)`**
with `r_coefficient ∈ RElement` over `coefficient_ring()`, so that in
the R-form view `L_label = r_coefficient · L_{section_rep}`.  (The
older signature `-> tuple[Label, tuple[int,...]]` was abelian-only;
the live contract returns a full `RElement` so non-abelian flavour can
contribute *virtual characters* — see the flavour section.)  For
unflavoured realisations it returns `(label, R.one())`.

**Two faces of the trace.**  The contract deliberately exposes both
the trace `Tr(L_a)` (primitive #6) and the **trace pairing**
`I_{a,b} = Tr(L_{ρ(a)}·L_b) = Tr(L_b·L_{ρ⁻¹(a)})` (`inner_product` —
derived-with-default but **overridable**).  The pairing is the Schur
index with line insertions, i.e. the inner product of **Schur
quantization** (Goal 2.2), and concrete realisations often compute it
by sharper direct methods than multiply-then-trace (`BPSKAlgebra`: the
single-Habiro-path Schur formula; the urq-torus realisation: the
ρ-free §6b formula).  `verify_trace_pairing_faces` checks the two
faces agree (cyclicity in pairing form);
`verify_inner_product_consistent` checks an override against the
multiply-then-trace default — for an overriding realisation this
agreement is a genuine cross-validation, not a tautology (see
`repo_audit.md` on enforced vs emergent).

### Axiom 5 — ρ-equivariance of the trace, `Tr ∘ ρ = ⋆ ∘ Tr`, and where it is inherited (user, 2026-09-18)

(In the user's `K_𝖖-algebras` draft this is axiom K5 — see "Cross-reference
with the `K_𝖖-algebras` draft" at the end of this file for the full label map.)

Cite it by its name, **ρ-equivariance of the trace**; the number is a
convenience (CLAUDE.md "Cite by name, not by number").

**Statement.**  The trace is ρ-equivariant, with ρ acting on the coefficient
ring through its own restriction to the centre, which is the Z₊-ring `⋆`
(`kalgebra.md` "The two involutions"; `RElement.star`, `RPowerSeries.star`):

    Tr(L_{ρ(a)})  =  ⋆( Tr(L_a) )                                  (trace form)
    I_{b,a}       =  ⋆( I_{a,b} )                                   (pairing form)

`⋆` is *not* bar: `𝖖` is untouched (a truncated power series has no image
under `𝖖 ↦ 𝖖⁻¹`, and `RPowerSeries` deliberately has no `.bar()`).  At trivial
flavour `⋆ = id` and the pairing form says the Gram matrix is **symmetric**;
with flavour it is **Hermitian** under `μ ↦ μ⁻¹` — the reading `repo_audit.md`
A73 first recorded on `a3`.  Verifiers: `verify_trace_intertwines_rho_star`
(one index, the primitive) and `verify_pairing_rho_star_symmetric`.

**The two forms are equivalent** given axioms 3–4.  Pairing ⟸ trace:
`I_{b,a} = Tr(ρ(b)a) = Tr(ρ²(a)ρ(b))` (cyclicity) `= Tr(ρ(ρ(a)b))` (ρ an
automorphism) `= ⋆Tr(ρ(a)b) = ⋆I_{a,b}`.  Trace ⟸ pairing: set `a = 1`.
Extension from labels to arbitrary elements: on the **flavour-in-labels**
tiers (BPS charts, RG flows, the Abe tier) the structure constants lie in
`Z[𝖖^±]`, which `⋆` fixes, so both forms extend by plain linearity on the
Z-form — which is what the RG inheritance below needs, since `RG(a)·S_RG` is
a formal sum.  On the **flavour-in-coefficients** cone tier (a3/a5/a7,
a1d4/6/8) the structure constants are `R`-valued and the honest extension is
the ⋆-semilinear `ConeKAlgebra.rho_element`, `ρ(c·L_w) = ⋆(c)·μ^{δ(w)}·L_{ρ(w)}`;
the base `KAlgebra.rho_element` (coefficients carried unchanged) is the wrong
extension there, which is the whole mechanism behind the a3 row of the table.

**Relation to the other axioms.**  Cyclicity at `u = 1` already gives
`Tr∘ρ² = Tr`; axiom 5 is a *square root* of it — consistent (`⋆² = id`) and
independent content.  The contract already pinned the ρ↔⋆ compatibility on the
**input** side (`verify_embed_intertwines_rho`: `embed_R(r⋆) = ρ(embed_R(r))`);
this is the **output** side, and `TKAlgebra` carries the τ-analogue on the ring
(TR3, `verify_tau_R_commutes_with_star_basis`).  For it to transport through
`base_change` / `forget` / `lower_flavour`, `RingHom` must commute with `⋆` —
a property its docstring always required and `verify_commutes_with_star` now
checks (every shipped hom passes).

**Origin — inversion of the torus, seen three ways.**  On the quantum torus
`ρ_Q(γ) = −γ` and the flavour direction is central, so `f ↦ −f` *is* `⋆` and
the axiom is immediate.  Every other tier either inherits that or realises the
same inversion in its own frame:

| tier | status of `I_{b,a} = ⋆(I_{a,b})` | mechanism |
|---|---|---|
| quantum torus | immediate | `ρ_Q(γ) = −γ`, flavour central |
| `RGKAlgebra` (generic) | **structurally inherited** | bilinear expansion `I_{a,b} = Σ_{c,d}[RG(a)S]_c[RG(b)S]_d I^aux_{c,d}`; swap `a↔b`, relabel `c↔d`, coefficients in `Z[𝖖^±]` fixed by `⋆` ⟹ inherited from the auxiliary.  No ring mismatch: `coefficient_ring()` *is* the auxiliary's.  Certificate: `verify_rg_inherits_rho_star` (hypothesis on the FS supports + conclusion) |
| `BPSKAlgebra` | pairing form a **tautology of the route**; trace form **content-carrying** | Nahm-sum hook: `I_{a,b} = Σ_sec Σ_{f_a,f_b} μ^{−f_a+f_b}·(h_a·h_b)` with both slots' c-data built by the same `_get_c` and the Habiro product commutative, so `a↔b` negates the μ-exponent identically and changes nothing else — `verify_pairing_rho_star_symmetric` cannot fail there.  `⋆` fixes `𝖖`, so there is no "`𝖖`-half" to be emergent; limitation (E) is about whether the Nahm sum equals the literal `Tr(ρ(a)·b)`, a different statement (`verify_inner_product_consistent`).  The check with content on BPS is `verify_trace_intertwines_rho_star` (through `trace`, a different computation) |
| `AbeKAlgebra` (pure `WRQTorus`, and since 2026-09-19 the matter `MatterWRQTorus` — `matter_wrq_torus.matter_sector_weight`, `inner_by_sector`, routed by `GNAbeKAlgebra.inner_product`; `tests/test_matter_sector_pairing.py`) | **manifest** | `I_{a,b} = Σ_m [u⁰](M_m(u) f^a_m(1/u) f^b_m(u))` with `M_m` inversion-invariant — a theorem from the closed form of the sector measure for all `G` and `(G, N)` (`sector_measure_closed_form` / `matter_sector_measure_closed_form`, 2026-09-19), pinned by the exact certificates `verify_sector_measure_inversion_symmetric` (367/367) and `verify_matter_sector_measure_inversion_symmetric` (203/203, `⋆` included); see "AbeKAlgebra — the abelianized-presentation contract" |
| hand-built presentations (finite/cone zoo, samples, skein) | **genuine content** | nothing derives it.  Unflavoured entries: pentagon, heptagon, e6 pass (on e6 the ρ-paired seeds `T₂=T₆`, `T₅=T₇` are exact in the frozen table).  Flavour-in-coefficients cone entries (a3/a5/a7): the LABEL-form verifier fails on exactly the rays where label-form orthonormality fails — the tier's documented split between the bare label `rho` and the `μ^δ`-carrying `ConeKAlgebra.rho_element`; the ELEMENT form passes, so this is a label-level non-compliance (a unit-character section choice), not a trace-data defect (repo_audit A74 as sharpened 2026-09-18).  It does not see defects that live in `multiply` (A75) |

So under the `repo_audit.md` A.1 discipline a green check is *evidence* on the
Abe tier and the hand-built zoo, a regression guard on RG, and on BPS
content-carrying only in the trace form.  Every number in this section is
pinned by `tests/test_kalgebra_rho_star.py` (default run, or `--full` where
marked): pentagon and heptagon (identity + all rays; unflavoured, hence the
weak form `Tr∘ρ = Tr`); e6's ρ-paired seeds; the pure Abe classes with the
sector route asserted to be what runs; a flavoured non-BPS RG flow
(`GMatterOverPure` u(2)+1, `⋆`-sensitive) through the inheritance certificate;
genuinely non-abelian `⋆` at `GNAbeKAlgebra(u_n(1), (1,), nf=3)` (`R(U(3))`; until
2026-09-19 the type-A `UNNfKAlgebra(1,3)` over `R(SU(3))`), fundamental
`↔` conjugate, both forms with `⋆`-sensitivity asserted) and, under `--full`,
`roster('su2-nf2')` (`R(U(2))`, including the magnetic label where ρ moves
flavour charge through `det^{−D}`, both forms); and a3's label/element split.

**Payoff beyond checking.**  As rows `t_{ρ(a)}[k] − ⋆(t_a[k]) = 0` in the
trace bootstrap (`trace_uniqueness._equations`), the axiom identifies unknowns
across **ρ-orbits** rather than ρ²-orbits — roughly halving the free parameters
(Goal 2.12); not yet wired.

### Optional contract — cone filtration and ρ²-orbit canonicalisation

Beyond the seven required primitives, `KAlgebra` exposes optional
hooks a realisation *may* override (all have safe defaults; not part
of the minimal contract):

| member | role | default |
|---|---|---|
| `cone_data() -> ConeData \| None` | Sidecar describing the 𝖖-commuting-cone structure (multiplicative generators, cones, cocycle, cross-products, bijection to native labels).  See `cone_data.py`. | `None` |
| `_multiply_via_cone_data(a, b)` | Generic reducer delegating `multiply` to `cone_data().derived_multiply`; a cone-equipped subclass can set `multiply` to a one-liner over it. | raises if no `cone_data` |
| `rho_squared_is_identity() -> bool` | Structural flag; `True` ⟹ all ρ²-orbits are singletons and ρ²-canonicalisation is a no-op. | `False` |
| `_canonical_rho2_orbit_rep(label)` | Canonical representative of a label's ρ²-orbit (ρ²-cyclicity `Tr(ρ²x)=Tr(x)` is a general axiom). | orbit walk (bounded); algebras with *infinite* ρ²-orbits — U(1)-gauged Sqed1, U1Square, U1Hex — MUST override with a closed-form drift-quotient |

The `Element` type is a small dataclass for linear combinations
`Σ_a c_a(q) L_a` over `Z[q, q⁻¹]`, with `__add__` / scalar `__mul__`
/ `bar()` / equality.  The R-form view `ElementOverR` is the same
shape with `RLaurent` coefficients; it is *derived*, never primary.

### Z-form vs R-form — two dict shapes of the same data

The Z-form `Element` is the canonical surface and the **working
representation**: structure constants `C^c_{ab}` are integral Laurent
polynomials in `q` on *full* labels `(section, R-basis)` —
`Element = dict[(s,w), LaurentPoly]`.

The R-form `ElementOverR = dict[s, RLaurent]` is a **derived view**, not a
second presentation.  It holds the *same* sparse data — the finite multiset of
quadruples `(section s, irrep w, q-power, Z-coefficient)` — merely **regrouped
by section** (`{s: {q: {w: c}}}`) instead of by full label (`{(s,w): {q: c}}`).
So `to_R_form` / `from_R_form` is a pure **regroup/transpose, no arithmetic**,
whenever the label literally *is* `(section, R-basis)`: `to_R_form` reads the
lift coordinate (`_label_section_decompose`, the derived bridge over the
`r_label_decompose` primitive) and `from_R_form` re-composes it
(`r_label_compose`).

**Which to use is a performance choice, not a correctness one.**  Because the
two carry identical information, the **Z-form is the efficient representation
for everything the algebra does**: `multiply` (per basis-pair), `ρ` (label
permutation), `trace` / `inner_product` (per label) all key naturally on the
full label and use plain-`int` `LaurentPoly` coefficients.  The R-form pays a
regroup to build and carries heavier `RElement` coefficient objects (a dict +
rep-ring multiply per coefficient), so doing arithmetic *in* the R-form is
strictly more expensive for no gain.  **Reach for the R-form only when a clear
need arises** — its one genuine advantage is *locality*: it groups all of a
section's flavour content under one key, handy for **recognising flavour
enhancement** or pushing a whole section's coefficients through a ring hom at
once.  Core arithmetic stays on the Z-form, and `forget` / `lower_flavour` /
`base_change` all transport over the Z-form lift coordinate — they do **not**
route through the R-form.

**Consistency caveat — a documented danger, not a contract guarantee.**  The
R-form faithfully represents the algebra iff the conversion round-trips,
`from_R_form(to_R_form(L_a)) == L_a`, which needs
`r_label_compose ∘ r_label_decompose = id` together with a clean `(section,
single-irrep)` decomposition (`verify_section_is_single_irrep`).  That
round-trip is exercised inline in the realisation tests that use the R-form,
but it is **deliberately not enforced contract-wide** (there is no
`verify_R_form_roundtrip`).  A realisation that exposes the R-form is therefore
responsible for ensuring — and testing — its own round-trip; if it does not,
its R-form is not a faithful view and must not be relied on.

The four sample algebras (Trivial, QuantumTorusZ2, Sqed1, Pentagon)
each implement the six primitives directly (all trivial-flavour, so the
flavour-lift coordinate is the universal default).  None of them know
about BPS quivers, Habiro arithmetic, or the Schur prefactor; that
scaffolding lives in `BPSKAlgebra`.

## Flavour in `KAlgebra`: abelian vs non-abelian

> **⚠ UNDER RE-AXIOMATIZATION — Plan 32 (user, 2026-06-14: "physics rules").**
> What follows describes the *current* flavour encoding (free over `R(G_f)`;
> the unit/non-unit split).  Per the paper, flavour is being re-axiomatized
> around the **forgetful map** `K_𝖖[T; G_f] → K_𝖖[T]` (the rep-ring
> **augmentation** `χ_r ↦ dim r`) and the **canonical lift** of the unflavoured
> basis — the lift canonical *up to a 1-dimensional rep of `G_f`* (semisimple
> `G_f` ⇒ unique lift).  Consequences being implemented: every flavoured
> `KAlgebra` gains an **unflavoured scaffold** (= `forget()`) and is built as a
> lift over it; the **augmentation becomes a required `ZPlusRing` primitive**.
> Whether the "splitting off a **non-unit** character **is** canonical" claim in
> §1b below survives is the gating **decision D1**.  Read this section as the
> current state, not the target.  See
> `restructuring_plans/32_flavour_forgetful_lift/`.

A flavoured `K_𝖖`-algebra (Ambrosino–Gaiotto, §"K_𝖖-algebras") is
enriched to an algebra over `Z[𝖖,𝖖⁻¹] ⊗ R_{G_f}` for a reductive
flavour group `G_f` with representation ring `R_{G_f}`: the canonical
basis (still over `Z[𝖖,𝖖⁻¹]`) includes the characters `χ_r` of the
finite-dimensional `G_f`-irreps, `ρ(χ_r) = χ_{r^∨}` (rep-ring
duality), and the trace is enriched to `Tr : A_𝖖 → Z((𝖖)) ⊗ R_{G_f}`.

In the code `R_{G_f}` is `coefficient_ring()`, a **Z₊-ring** (Lusztig–
Ostrik) — see `zplus_ring.py`.  The flavour content of a canonical
label is exposed by the `_label_section_decompose` primitive, whose
`RElement` second component is the character `χ_r`.  The shipped rings:

| `ZPlusRing` | `G_f` | basis | multiplication | `⋆` (= `χ_r ↦ χ_{r^∨}`) |
|---|---|---|---|---|
| `TrivialZPlusRing` | trivial | `{()}` | trivial | id |
| `AbelianZPlusRing(n)` | `U(1)ⁿ` (max. torus) | `Zⁿ` (`μ^f`) | **group-like** `μ^f·μ^g = μ^{f+g}` | `f ↦ −f` |
| `SU2ZPlusRing` | `SU(2)` | `ℕ` (spins) | **Clebsch–Gordan** (step 2) | id (self-dual) |
| `SO3ZPlusRing` | `SO(3)` | `ℕ₀` (int. spins) | **Clebsch–Gordan** (step 1) | id (self-dual) |
| `SU3ZPlusRing` | `SU(3)` | `ℕ²` (Dynkin) | **Klimyk + Freudenthal** | `(p,q) ↦ (q,p)` |
| `SU4ZPlusRing` | `SU(4)` | `ℕ³` (Dynkin) | **Klimyk + Freudenthal** | `(p,q,r) ↦ (r,q,p)` |
| `SUNZPlusRing(N)` | `SU(N)`, any `N` | partitions, `< N` rows | **Littlewood–Richardson** (Kostka-DP) | conjugate partition |
| `SO2NfZPlusRing(Nf)` | `Spin(2Nf)=D_Nf` | dom. wts (int [tensor] / half-int [spinor]) | **Clebsch–Gordan** (Weyl-denom. un-branch) | `w_0` (self-dual even `Nf`; spinor-swap odd) |
| `SU2xU1ZPlusRing` | `SU(2)×U(1)` (= U(2)) | `(k,m)` (spin k/2, U(1) m) | **Clebsch–Gordan** ⊗ group-like `μ` | `(k,m) ↦ (k,−m)` |

All ship in `zplus_ring.py` except `SO2NfZPlusRing` (`so2nf_characters.py`).
The bespoke `SU2`/`SU3`/`SU4` rings (spin / Dynkin labels, with
`to_abelian`/`from_abelian` torus embeddings) are the certified low-rank
twins of the general `SUNZPlusRing` — `SUNZPlusRing(2)≅SU2`,
`(3)≅SU3`, `(4)≅SU4` are pinned exactly (incl. `⋆`) in
`tests/test_sun_zplus_ring.py`; they are kept because their label
conventions and torus maps are what individual theories consume.  A
*tensor-product* helper also ships — `tensor_zplus_ring.TensorZPlusRing`
(`R₁⊗R₂`, pairwise basis; the former duplicate `ProductZPlusRing` was
**unified into it**, 2026-06-13); the
SU-flavour multiplet rings `sun_characters.SUNFlavourRing` (`⊗ R(SU(M_i)) ⊗
R(U(1)^r)`) and the enhancement wrappers (`flavour_enhancement`,
`sun_flavour_enhancement`) un-branch a Cartan presentation to its genuine
non-abelian flavour ring.

The abelian and non-abelian cases differ in ways that are easy to get
wrong, so they are spelled out below.

### 1. Group-like (abelian) vs fusing, non-invertible (non-abelian) characters

In the **abelian** ring `R(U(1)ⁿ) = Z[μ^±]` the characters are
*group-like and invertible*: a label's flavour content is a single
`μ`-monomial, and `_label_section_decompose` returns
`(section_label, R.basis_element(flav_coords))`.  This is what a
BPS-quiver realisation produces by default — `Γ_f = ker(B)` is split
off by SNF and the flavour coordinates become the `μ`-exponents
(`bps_kalgebra.py`, `quantum_torus_kalgebra.py`).

In a **non-abelian** rep ring, irrep characters *fuse* — products are
non-negative sums of irreps (Clebsch–Gordan / Klimyk), and a
higher-dimensional `χ_r` is **not invertible**.  The non-abelian ring
sits inside the abelian one only as the **Weyl-invariant subring**
(`χ_j ↦ μ^j + μ^{j-1} + ⋯ + μ^{-j}`, via `SO3ZPlusRing.to_abelian`
etc.).  Consequently a single canonical label can carry a *virtual
character* — a **signed** sum of irreps.  The worked case is the
half-integer-spin sector of an SO(3)-flavoured algebra
(`so3_quantum_torus_kalgebra.py`, `so3_bps_kalgebra.py`): an even
σ-orbit chain has a w-fixed midpoint and decomposes to a single
`χ_j`, but an **odd** chain has no fixed midpoint and decomposes to

```
    T_j(χ_·) = Σ_{i=0}^{j} (−1)^{j−i} χ_i        # a virtual character
```

solving the half-integer recursion `w_{j+½} = χ_j · w_½ − w_{j−½}`.
Because such labels are not group-like, the **section choice is not
unique** for half-integer spins (the anchor is fixed only up to a
gauge equivalence); abelian SNF sections, by contrast, are canonical
given the section basis.

### 1b. Freeness over `R` — a convention, not a contract (the tensor recipe)

A flavoured `A_𝖖` is always free over `R = coefficient_ring()`, but
freeness is **mathematically convenient, not physical**, and in general
**not canonical** (an R-basis is a choice of section; two sections
differ by *unit* characters).  So the contract **does not encode it** —
there is no `free_over_R` flag, and `multiply` is **always Z-valued**
(`LaurentPoly` on canonical labels, flavour in the labels; `RLaurent`
appears only in the *derived* `to_R_form` view and Layer-1 / trace
auxiliaries, never at the `multiply` boundary).  A realisation that
finds the R-form convenient simply implements the optional hooks
`section_decompose` / `embed_R` (and gets `to_R_form` / `from_R_form` /
`rho_R_form` for free).

**The unit boundary.**  Whether you can canonically split off a
character is decided by *invertibility* (§1).  Unit characters (abelian
`μ^f`) are a **torsor** — `μ^f·M_s = M_{s+f}` is just another section,
so "factoring them out" is a non-canonical choice; they stay
**flavour-in-labels** (a charge coordinate).  Non-unit (fusing)
characters (`SU(2)` `χ_w`, w≥1) are genuinely independent free-module
generators — splitting them off **is** canonical.  So: *free over the
non-unit characters; the torus stays in the labels.*

**The tensor recipe.**  When `R = R₁ ⊗ ⋯ ⊗ R_k`, this generalises
factor-by-factor.  Choose a *free* sub-product `R_free = ⊗_{i∈S} R_i`
and a *label* sub-product `R_lab = ⊗_{i∉S} R_i`.  Then:

* `section_decompose(label)` peels the `R_free` character as the
  `r_coeff`; the `R_lab` charge stays in the section representative.
* `embed_R` is the central embedding **on `R_free`** (a map of canonical
  bases, `χ_w ↦` a central canonical element, extended by fusion); on
  `R_lab` a *unit* character acts as a **section charge-shift** (a single
  relabelled basis element, coefficient 1).
* `multiply` stays Z-valued: run the internal section product, fuse the
  input characters (`RElement.__mul__` multiplies factor-wise), and on
  re-expanding route the **`R_free` part → label character coordinate**
  and the **`R_lab` part → section charge**.

Worked examples:

| algebra | `R` | `R_free` | `R_lab` |
|---|---|---|---|
| octagon, a3/a5 | `R(U(1))` | trivial | `R(U(1))` (all in labels) |
| a1dodd, su3 | `R(SU(2))`, `R(SU(3))` | all of `R` | trivial (fully free) |
| a1deven | `R(SU(2))⊗R(U(1))` | `R(SU(2))` | `R(U(1))` |

`finite_a1d3_zform.py` (`FiniteA1D3ZKAlgebra`) is the fully-free case
worked end-to-end: labels `(w, section)`, `multiply` Z-valued via
internal section product + Clebsch–Gordan fusion, validated isomorphic
to the native R-form.

**Structured home: the `RKAlgebra` specialisation.**  *(⚠ Plan 32 A4 / retire,
2026-06-14: `RKAlgebra` was **decoupled** — its 3 production subclasses are
standalone `KAlgebra` / `ConeKAlgebra` with the fusion copied in — and then
**retired to `legacy/rkalgebra.py`** (production-orphaned).  The flavour
sharpening runs on the `KAlgebra` surface, not through `RKAlgebra`; the section
below is **historical context** for the freeness encoding.)*  The base
`KAlgebra` stays minimal (freeness undeclared); a realisation that wants
freeness *structured* — rather than hand-writing the two hooks — can
subclass `RKAlgebra(KAlgebra)` (`rkalgebra.py`), which **specifies a ring
it is free over (`R_free`) and, optionally, a ring it is not (`R_lab`)**,
with `coefficient_ring() = R_free ⊗ R_lab` (`tensor_zplus_ring.py`;
just `R_free` when `R_lab` is trivial).  Canonical labels are `(w, s)` —
an `R_free` character and a *section* (carrying any `R_lab` charge).  The
realisation supplies only its **section engine** (`section_multiply`,
`section_identity`, `section_rho`/`section_rho_inverse`, and — for
non-trivial `R_lab` — `shift_section`); `RKAlgebra` *derives*
`multiply` (Z-valued), `rho`, `section_decompose`, `embed_R` from the
free/label split.  `FiniteA1D3ZKAlgebra` is the fully-free instance
(`R_lab` trivial); the SU(2)×U(1) (a1deven) shape is exercised by the
toy in `tests/test_rkalgebra.py`.  `SU3ADKAlg` (`su3_ad_kalg.py`) is the
real fully-free SU(3) example — it inherits **both** `RKAlgebra` and
`ConeKAlgebra` (MRO `[SU3ADKAlg, RKAlgebra, ConeKAlgebra, KAlgebra]`):
`RKAlgebra` derives `multiply`/`rho`/`embed_R` from the section engine
(its 5-tuple labels `(tile,a,b,p,q)` map to `(free_char=(p,q),
section=(tile,a,b))` via `_pack_label`/`_unpack_label`), while
`ConeKAlgebra` keeps `cone_data` (the section engine) + the trace
pipeline + the `isinstance` identity.  This mirrors how `TKAlgebra` /
`RGKAlgebra` extend `KAlgebra` — freeness is a *specialisation*, not a
base-contract obligation, and it composes with `ConeKAlgebra` by C3
linearisation (common `KAlgebra` base).

### 2. The two involutions: bar (`𝖖`-only) vs `⋆` (rep-ring duality)

`R[𝖖^±]` carries **two genuinely different involutions that must not
be conflated** (`zplus_ring.py` is emphatic about this):

* the `K_𝖖`-algebra's **bar** acts *only on `𝖖`* (`𝖖 ↔ 𝖖⁻¹`),
  leaving `R` untouched — structure constants are palindromic in `𝖖`
  with the `μ`-dependence transparent (`RLaurent.bar()`);
* the Lusztig–Ostrik **`⋆`** is rep-ring duality `V ↦ V*` on `R`
  itself — `μ^f ↦ μ^{-f}` (abelian), id (SU(2)/SO(3), self-dual),
  `(p,q) ↦ (q,p)` (SU(3)) — and is **`ρ` restricted to the central
  flavour subalgebra**: in any QT realisation `ρ(X_γ) = X_{−γ}`, so on
  `μ^f = X_{γ_f}` we get `ρ(μ^f) = μ^{−f}` (`RElement.star()`,
  `ZPlusRing.star_basis`).

So `ρ` is `⋆`-twisted-linear on the R-form (`ρ(c·u) = c.star()·ρ(u)`)
even though it is plain-linear on the Z-form — the twist lives
entirely in the centre and surfaces only through `to_R_form` / the
R-side of the trace.  Reaching for `bar()` where `⋆`/`ρ` is meant (or
vice versa) is the most common flavour bug.

### 3. Orthonormality with a non-abelian `R`

The paper's flavoured orthonormality is on the **identity summand**:
`I^{(1)}_{a,b} = δ_{a,b} + O(𝖖)` (the `χ_0 = 1` component of `I_{a,b}
∈ R((𝖖))`), *not* the whole `R`-valued series.
`KAlgebra.verify_orthonormality` implements exactly this (2026-06-10):
it reads the `χ₀`-component of the `𝖖⁰` coefficient and compares to
`δ_{a,b}` — which is faithful on canonical labels understood as pairs
`(section label, R element)`: by Schur orthogonality the
`χ₀`-component of `r_a⋆ · r_b` is `δ_{r_a, r_b}`, abelian and
non-abelian alike.  (Companion requirement: `rho(label)` must be the
*canonical-label* action — `⋆` on the central R-part — not a raw
chart map; see `BPSKAlgebra._sec_rectified_map` and `repo_audit.md`
finding A8.)  With non-abelian `R` the trace coefficients are
themselves (possibly **signed**) combinations of irrep characters, so
"`δ_{a,b}`" means the multiplicity of the *trivial* rep `χ_0`; for the
self-dual SO(3)/SU(2) rings the trace on a `χ_j` label returns the
single character `χ_j` (no sign), and the identity summand is read off
as its `χ_0`-component.

### 3b. Section-orthonormality axiom + the unflavoured specialisation (Plan 32)

The **unflavoured canonical basis is the section image** `{M_s}` (= the
`forget()` image): `forget = base_change(ε)`, `ε = coefficient_ring()
.augmentation()` the rep-ring augmentation `χ_r ↦ dim r` (a `ZPlusRing`
primitive, Plan 32 T1).  A flavoured canonical `L_a` with
`_label_section_decompose(a) = (s, χ_r)`:

- **dressed by a 1-dim rep** (`r ∈ Λ`, `dim 1`) ⟹ `forget(L_a) = M_s` lands
  **on** an unflavoured canonical (norm 1) — a section up to the `Λ` torsor;
- **dressed by `χ_w`** (`dim > 1`) ⟹ `forget(L_a) = dim(w)·M_s`, a **multiple**
  (norm `dim w ≠ 1`), not canonical.

So the sharp orthonormality axiom is **on the sections**: for sections `s, s'`,

    I_{s, s'} = δ_{s, s'} + O(𝖖)        (the unflavoured Goal 2.1, on `{M_s}`).

For dressed canonicals (`χ_w`, dim>1) the `χ₀`-component still reads `δ` —
`I_{a,b}` carries `χ_{w^∨}·χ_w = χ_0 ⊕ (rest)`, containing **exactly one**
identity copy (`mult_triv(V_w^*⊗V_w) = dim Hom(V_w,V_w) = 1`) — but the *full*
`R`-valued leading term `ε`-augments to `(dim w)²`.  (Worked: SU(3) fundamental
`I_{d,d}|_{𝖖⁰} = χ_{(0,0)} ⊕ χ_{(1,1)}`, `ε = 1 + 8 = 9`.)

The unflavoured scaffold is materialised by **`KAlgebra.forget() -> KAlgebra`**
(`kalgebra.py`): `base_change(ε)` *plus* the collapse of the flavour-in-labels
onto sections.  `_ForgetKAlgebra` **extends `_BaseChangeKAlgebra`** to add that
collapse — `base_change(ε)` alone is *coefficient-only* and keeps the flavoured
labels, so it is **not** the unflavoured algebra.  `forget()` is a genuine
`KAlgebra` over `Z` on the section basis: `forget().multiply` is the
`dim`-weighted unflavoured product, `forget().trace`/`inner_product` are `ε∘(…)`,
with `forget().inner_product(s, s') = δ + O(𝖖)` on sections (the unflavoured
orthonormality) and `(dim w)²` on a `χ_w`-dressed canonical.  Default: `self`
if already unflavoured, else `_ForgetKAlgebra(self)` — the unflavoured KAlgebra
is **produced on demand**, not carried.  A subclass may override only with a
genuine native realisation of the *same theory with `G_f` forgotten* (not a
flow's pure-gauge base, which is a different theory).  Tests:
`tests/test_unflavoured.py`.

**Partial forgetfulness — `lower_flavour(φ)`.**  Replacing `ε` by any ring hom
`φ` compatible with augmentation (`ε_target∘φ = ε_source`, carrying `Λ → Λ'`)
*lowers* `G_f` instead of forgetting it: `KAlgebra.lower_flavour(φ)`.  The
transport is uniform — **identify sections and push the R-elements through `φ`**:
each canonical `(s, χ)` goes to `(s, φ(χ))`, re-expanded over the `R'`-irreps
(`_LoweredFlavourKAlgebra` extends `_BaseChangeKAlgebra` with the section-engine
transport; labels are `(section, R'-irrep)`).  Unlike the `φ = ε → Z` **merge**
(`χ_r·M_s ↦ dim(r)·M_s`, = `forget()`), a `φ` that grows `Λ` (the 1-dim reps)
**splits** a canonical: SU(2)→U(1) sends `χ_j·M_s ↦ Σ_k μ^k·M_s`, so one SU(2)
canonical becomes `2j+1` U(1) canonicals — `dim` is preserved by the
augmentation-compatibility.  It needs only `dim` (classify units) — no `recognize`
primitive.  `trace` transports too (it is R-linear over the centre,
`Tr((s,χ)) = χ·Tr(M_s)`, so `Tr'((s,w)) = w·φ(Tr(M_s))`), and `inner_product`
follows — the lowered algebra is orthonormal.  Validated
`tests/test_lower_flavour.py`: `φ=id` → iso to source, `φ=ε` → `forget`,
SU(2)→U(1) → split (valid KAlgebra, trace + orthonormality).

### The free-R-module encoding: `section_decompose` + `embed_R`

A flavoured `A_𝖖` is **free as a module over `R`, but not canonically**:
an R-basis is a *choice of section* (one representative per central-R
orbit), and two sections differ by multiplication by **unit characters
`μ^f`** — a torsor, with no canonical R-basis.  The Z-form basis `{L_a}`
is canonical but doesn't exploit the R-module economy; the R-form
(section) basis is economical but non-canonical.

This is encoded faithfully by **two** pieces of independent data (and
*only* two — everything else is derived):

| member | meaning |
|---|---|
| `section_decompose(a) → (section, r_coeff)` | the R-module coordinate of `L_a` (public face of `_label_section_decompose`); `r_coeff` a single character (abelian) or a virtual character (non-abelian) |
| `embed_R(r) → Element` | the central embedding `ι : R ↪ A_𝖖`, a **map of canonical bases** `χ_w ↦ χ_w`; default embeds `1_R ↦ identity`, flavoured realisations override |

bound by the **faithfulness axiom** `embed_R(r_coeff(a)) · L_{section(a)}
== L_a` (verifier `verify_embed_section_roundtrip`), and the
compatibility `embed_R(r⋆) == ρ(embed_R(r))` (verifier
`verify_embed_intertwines_rho`, since `ρ` restricts to `⋆` on the
centre).

Everything section-relative is then **derived**, never a primitive:
the lift `from_R_form(x) = Σ embed_R(c_o)·L_{s_o}` (so the non-abelian
"is `R·L` canonical?" question is dissolved — `multiply`/fusion answers
it), and `rho_R_form = to_R_form ∘ rho_element ∘ from_R_form` (the
`⋆`/`δ` torsor corrections fall out automatically — no bespoke
`rho_delta`).  The whole API is **additive**: `embed_R` is concrete
with a default, `section_decompose` a public delegator, so no existing
realisation breaks (see `restructuring_plans/18_rho_rform_placement/`).

### Base change across the abelian / non-abelian boundary

`base_change(phi)` pushes the trace's R-coefficients through a Z₊-ring
hom `phi : R_1 → R_2` (physically, flavour-symmetry restriction
`α : H → G`, `phi = α^*`).  Concrete factories in `zplus_ring.py`:
`identity_hom`, `augmentation_hom` (`μ → 1`, gauge quotient),
`restriction_hom` (torus → subtorus), and the non-abelian bridges
`so3_to_u1_hom` (`R(SO(3)) → R(U(1))`, restrict to the maximal torus),
`so3_to_su2_hom` (even-spin inclusion), and the helper `u1_weyl_to_so3`
(Weyl-symmetric `μ`-Laurent → SO(3) characters).

**The inverse direction — flavour *enhancement*.**
`flavour_enhancement.FlavourEnhancementKAlgebra(base, Nf)` *un-branches* a
`U(1)^Nf`-Cartan-flavoured base (`base.add_flavour(Nf)`) up to the non-abelian
`Spin(2Nf)` rep ring `so2nf_characters.SO2NfZPlusRing(Nf)` — the exact inverse
of `base_change(restriction_hom)`, well-defined on the `W(D_Nf)`-invariant
part by `R(Spin(2Nf)) ≅ R(T)^{W(D_Nf)}` (recognizer
`verify_flavour_enhancement`).  Worked case: **SU(2)+Nf** (flavour `SO(2Nf)`)
— `Spin(4)=SU(2)²` (`Nf=2`), `Spin(6)=SU(4)` (`Nf=3`), `Spin(8)` triality
(`Nf=4`).

### Current scope

Non-abelian flavour is implemented for **one simple factor**:
`SU2ZPlusRing` requires `rk ker(B) = 1` (a single SU(2) factor — the
finite cover of SO(3); the SO(3) realisations were retired 2026-06-14,
Plan 32 A5, with the half-integer sector now an honest single SU(2)
irrep); `SU3ZPlusRing` requires `rk ker(B) = 2` with `σ` a pure 3-cycle
on `Γ^{σ≠1}` (one SU(3) factor of pure flavour).  Multiple / mixed
non-abelian factors are future work.

## Charges vs labels — three roles of an integer tuple (read this first)

> **Symbol conventions (user ruling, 2026-07-29) — two collisions, resolved.**
> **`ρ` is the canonical algebra automorphism and nothing else**; the Weyl vector
> is not `ρ`.  Write the **sum of the positive roots** directly,
> `Σ⁺ := Σ_{α>0} α` (the Weyl vector being `½Σ⁺`), so that `⟨Σ⁺, α_i^∨⟩ = 2`,
> root height is `⟨Σ⁺, m⟩`, the parity character is `π = ⟨Σ⁺,·⟩ mod 2` and the
> atom phase is `𝖖^{½⟨Σ⁺,m⟩}` — with the `½` visible.  That `½` is real but is **not
> an obstruction** (Plan 24 D31): the algebra needs the phase only through its
> coboundary `δS̃`, an integer even where `S̃` is not, since `π` is Weyl-invariant
> *and* additive.
> **`σ` is a BPS notion only and has no meaning outside a `BPSKAlgebra`**: it is
> the half-monodromy, the label-level `ρ` on BPS canonical labels, piecewise-linear
> (see the table below).  The **Dynkin diagram involution is `−w₀`**, never `σ`;
> an unrelated leaf-exchange Poisson involution on a `ZPlusRing` is `ε`.
> **Charge tuples are written `(m, e)` — magnetic charge first, always** (user
> ruling, 2026-09-21: *"we always use an (m,e) convention, not (e,m)"*): the
> Kapustin labels `(m, e)` / `(m, λ)`, the sample algebras' `(m, n)`, and a
> gauge-theory BPS chart's lattice `Z²(m, e) × Λ` with `⟨(1,0),(0,1)⟩ = +1`
> (the draft's `⟨(1,0,0),(0,1,0)⟩ = 1` for SQED_2).  The one module that had
> built its lattice as `(e, m)` — `implementations/gf_bps_sqed_nf.py` — was
> converted that day (Plan 27 `decisions.md` A8).

A `BPSKAlgebra` involves **two different algebras** over `Z[𝖖^±]`, both
with `Z^n`-valued labels, related by the RG map `F`.  An integer tuple
can therefore mean three things, and conflating them is the single most
common bug in this codebase.

| symbol | object | algebra | `ρ` acts by | legal operations |
|---|---|---|---|---|
| **`γ ∈ Γ`** — *charge / QT-index* | `X_γ` | the **auxiliary** `QuantumTorusKAlg(Γ, B)` | `ρ_Q(γ) = −γ` (linear) | `γ + γ'`, `−γ`, `⟨γ, γ'⟩` |
| **`a, b, c`** — *canonical label* | `L_a` (= `F(L_a)`) | the **`BPSKAlgebra`** `A_𝖖[T]` | `σ`, the half-monodromy (**piecewise-linear**) | `multiply(a, b)` **only** |

The "charge" and "QT-index" are the *same* data — the quantum torus is
literally indexed by `Γ`, so a `Γ`-charge `γ` **is** a genuine *label*
of the QT (the canonical-basis label of the generator `X_γ`), not merely
an abstract charge.  **Convention (repo-wide for BPS docs): `γ` always
denotes a charge / QT-label; `a, b, c` always denote BPS canonical
labels.**

The two are bridged by `F = RG`, the **RG-flow algebra homomorphism**

```
F : A_𝖖[T] → Q_𝖖(Γ),   F(L_a) = dict{ γ (QT index): c_γ(𝖖) ∈ Z[𝖖^±] }
```

It is an **algebra map** — `F(L_a · L_b) = F(L_a) · F(L_b)` in the
quantum torus (the `verify_rg_multiplicative` axiom on `RGKAlgebra`) —
which is exactly how structure constants are computed: multiply
`F(L_a) F(L_b)` in the *easy* QT and read off the `F(L_c)`.

**Discovery vs abstraction (why there are two notations).**  Intuitively
the `BPSKAlgebra` first *discovers* a canonical element `F_γ` from a
charge `γ` via the defining relation

```
F_γ · S = X_γ + O(𝖖)          # S = the spectrum generator
```

(`F_γ` is the unique element whose `F_γ S` has leading QT term `X_γ`),
and *then* treats it as the RG image `F_{γ(a)} = F(L_a)` of an abstract
algebra element `L_a`.  So `F_γ` (subscript by a **charge** `γ` — the
computational/discovery view) and `F(L_a) = F(a)` (the abstract image of
the **label** `a`) are the *same* object, with `γ = γ(a)` its tropical
charge.  **Write `F_γ` (charge subscript), `F(a)`, or `F(L_a)` — never
`F_a`: a label has no business as a subscript on `F`.**

**Lower and upper tropical labels.**  The charge identifying `a` is
precisely its **lower tropical label** `γ₋(a)`: by default `a` *is*
`γ₋(a)` (the BPS canonical basis is indexed by its lower tropical
charges), and `F_{γ₋(a)} · S = X_{γ₋(a)} + O(𝖖)`.  There is also an
**upper tropical label**

```
γ⁺(a) = ν_S(γ₋(a)) = −σ⁻¹(γ₋(a))          # σ = the half-monodromy (label-level ρ)
```

and `F(L_a)` is supported on the **doubly-tropical interval**
`[γ₋(a), γ⁺(a)]` (in cone order) — exactly the support the F-solver
enumerates.  So one can write `γ(a)` for the tropical charge of `a`,
meaning `γ₋(a)` unless the upper `γ⁺(a)` is named explicitly.

**DON'Ts (these are the actual bugs):**
- Do **not** add or negate canonical labels: `a + b`, `−a`, `⟨a, b⟩`
  are meaningless, and `multiply(a, b) ≠ L_{a+b}`.  Those operations
  belong to charges `γ` in the QT.
- Do **not** apply `ρ_Q(γ) = −γ` to a canonical label.  On labels `ρ = σ`
  is **piecewise-linear** (see the "ρ is piecewise-linear, NOT Z-linear"
  alert above) — *not* negation.
- The keys of `F(a)` are **QT indices `γ` (full `Γ`-tuples)**, not
  canonical labels.
- **Flavour conventions differ between the two algebras.**
  `QuantumTorusKAlg` labels are **full `Γ`-tuples** (`X_γ` and
  `X_{γ+γ_f}` are *distinct* for `γ_f ∈ Γ_f`); `BPSKAlgebra` canonical
  labels are **`Γ_g` section reps**, with the flavour shift absorbed
  into a μ-coefficient.  So "the label `γ`" means different tuples on
  the two sides — naive `verify_*` comparisons *across* the RG map can
  mismatch purely from this convention, not from a real error.

## `RGKAlgebra` and `BPSKAlgebra` — the relation (read before adapting subclasses)

Since Plan 20, `RGKAlgebra` is a **complete `KAlgebra`** that derives the
whole contract generically from RG data; `BPSKAlgebra(RGKAlgebra)` is the
quantum-torus realisation.  A `BPSKAlgebra` *is* a generic
`RGKAlgebra`-over-its-quantum-torus, decomposed as follows (audit:
`restructuring_plans/20_rgkalgebra_functional/`):

1. **Optimized overrides of contract methods (same math, BPS faster).**
   - `RG(a) = F(a)` — `F` is simply BPS's name for `RG`.
   - `multiply`, `trace` — *already the same algorithm* as the generic
     `RGKAlgebra` (`from_ir_image(RG(a)·RG(b))`, RG-transport trace);
     differences are caching only.
   - `rho`/`rho_inverse` — same maps; BPS computes them as the closed-form
     half-monodromy `σ` (from the spec), where the generic path runs the
     `solve_rg` mirror / the `tRG` opposite-algebra solve.
   - **Why BPS is sharply faster:** the quantum torus has `deg(X_γ) = γ`
     an *isomorphism* — every graded piece is **one-dimensional**
     (`label ↔ charge`).  That bijectivity is what enables scalar
     arithmetic, the doubly-tropical bound, and the `σ` closed form.  It
     is specific to the QT realisation almost by definition; a generic
     `RGKAlgebra` over a multi-dimensional-graded auxiliary cannot use it.

2. **Trivial `RGKAlgebra` wiring.**  `grading()` = `(Γ, deg(X_γ)=γ,
   height=central charge)`.  **Flavour is delegated to the shared,
   flavour-aware quantum-torus auxiliary** (`coefficient_ring`, the
   section/`ker B` split, and the `δ_{[γ],0}·μ^{flav}·(q²)_∞^{rk Γ_g}`
   trace) — not a BPS-specific structure.  The **tropical labels**
   `gamma_lower`/`gamma_upper` are a generic readout: the smallest /
   largest charge in `RG(a)`'s support.

3. **Truly unique to `BPSKAlgebra` (no generic analogue).**  The **BPS
   quiver + spec** (quiver node charges → the positive cone; spec → both
   `S_RG` and `σ`); the chart tooling (`spectrum_generator`, `root_data`,
   `shorten_spec`, node-deletion RG flows); and — the headline —
   **MUTATIONS**: cluster-mutating the quiver/spec yields *another*
   `BPSKAlgebra` (chart) for the **same abstract `KAlgebra`**, an
   often-infinite family of equivalent presentations linked by explicit
   `KAlgebraIso`s (`chart_graph.py`, `bpskalgebra_iso.py`).  This is the
   IR cluster description (Goal 3) and the "one object, many
   presentations, certified equivalent" frame (Goal 1.3).

**Adapting a new RG-flow theory to `RGKAlgebra` — the standard recipe.**
Supply only the **flow data** — `auxiliary()` (a graded complete KAlgebra,
often `base.add_flavour(R)`), `grading()`, and the spectrum generator
`_s_rg_component()` / `rg_generator()` (+ `apex()` if non-identity) — and
**inherit the whole API** (`RG`, `multiply`, `trace`, `inner_product`, `ρ`).
The matter/flavour dressing lives **in `S_RG`** (it surfaces through
`RG(a)·S_RG`, consumed by the generic multiply/trace), *not* in overrides:
override a primitive only for a genuinely sharper closed form (as
`BPSKAlgebra` does with `σ`), **never to repair a mis-specified flow**.  If
`RG` comes out trivial, the index wrong, `rho_inverse` raises / regrows the
registry, or the trace is pathologically slow, the flow data is wrong — fix
it, don't paper over it with overrides (the #336→#337 lesson).

**Copy the closest of the many worked flows** rather than improvising:
- *matter over pure gauge* (`add_flavour(R)` + `S_RG = Ψ = ∏ E_𝖖(μ·v)`):
  `u1_nf_rgflow`, `su2_nf_over_pure_rgflow`, `un_nf_over_pure_rgflow`
  (the most complete: chart-evaluated §6b trace/inner-product override —
  per-flavour-level URQTorus FS images, the sanctioned speed-only
  pattern — plus the certified flow↔BPS `KAlgebraIso` at U(2)+N_f=1,
  `un_nf1_over_pure_iso`; matter normalization in
  `un_nf_matter_dressing.md`),
  `su2su2_bifund_over_pure`, `su2_linear_quiver_over_pure`,
  `quiver_over_pure` (the generic linear-quiver class).  The **correct
  flavour rings** for these (D5, user 2026-06-11) are `SU(N_f)` /
  per-node `SU(M_i)` — the diagonal / link `U(1)`s are gauge-centre
  bookkeeping; present them via `sun_flavour_enhancement.enhance_un_nf`
  / `enhance_quiver` (general `SUNZPlusRing` in `sun_characters.py`; the
  type-A analogue of the `Spin(2Nf)` un-branching machinery in §"Base
  change across the abelian / non-abelian boundary");
- *AD chains* (drop BPS-quiver nodes → flavour): `a1a2k_rgkalgebra` (the
  generic-finder exemplar: `cone_gens` + `_s_rg_component`),
  `a1aodd_to_even_rgkalgebra` (U(1) flavour), `a1deven_rgkalgebra` /
  `a1deven_to_even_rgkalgebra` (U(2) flavour);
- *BPS node-drop* (the standard case): `directional_subquiver_rg`
  (`DirectionalSubquiverRG` / `DirectionalSingleNodeRG` — see the
  dedicated section below; `rg_flow.SubquiverRG`/`SingleNodeRG` are now
  thin constructors over it);
- *minimal* (`auxiliary` + `grading` + `S_RG` over a quantum torus):
  `_PentagonViaQT` (`tests/test_rgkalgebra_graded.py`);
- *composition / factorization*: `ComposedRG` (`then`), `ExtractedRG`
  (`factor_through`), `ConeRGKAlgebra`.

**Knowing `S_RG` — two independent contracts (not interderivable).** The
spectrum generator is supplied two ways, for two different consumers:

- **`_s_rg_component(p) → {label: HabiroElement}`** — the **exact**
  `Γ_RG`-graded component `[S_RG]_p`: never q-truncated, finite (`≤ h(p)`
  spec factors, by height-positivity), `{}` off the cone.  This is
  "knowing `S_RG` exactly", and is what **RG solving** consumes
  (`solve_rg_exact`, the per-charge oracle path) — the peel walks *up the
  grading-height cone*, so it requires each `[S_RG]_p` exactly,
  height-graded.
- **`rg_generator(cutoff)`** — `S_RG` windowed by **q-order ≤ cutoff**.
  `S_RG` *knowledge* (the windowed-solve fallback consumes it; the generic
  `trace` assembles its object from it).  It is **not**, by itself, the
  right trace handle: a `q^K` trace needs the **product** `RG(a)·S_RG` to
  q-order `K` (`rg_times_s_rg`, the safe trace primitive), not `S_RG` to
  q-order `K`.

They are *not* interderivable: `q-order(δ) = h(δ) + binding(δ)`, where
`binding` is the Dirac-pairing **quadratic** form (`≥ 0` on the cone, `0`
on the rays).  So q-order is not a grading height, the charge↔q-order map
is subclass-specific and non-monotone in `h`, and a q-order window slices
the cone wrong — conflating the two is what let the heptagon bound state
slip in #309 (fixed in #310).  *Status:* both contracts are live.
`trace` / `inner_product` consume the **FS object** `rg_times_s_rg(a, K)` =
`RG(a)·S_RG` to q-order `K` (the safe, overridable trace primitive; the
generic default assembles it from the q-windowed `rg_generator` at the fixed
`_rg_cutoff()` — a heuristic, sound for the tested small-`K` regime, which a
flow can override for certified traces).  **RG solving consumes the exact
oracle** — `RG`/`tRG` route to
`graded_rg_solver.solve_rg_exact`, which fetches `_s_rg_component(γ)` on
demand and computes every residual exactly (no `S_RG` window, no mirror —
past the true top the residual is exactly `O(q)`, so the cone-walk
terminates on its own), whenever the grading carries `cone_gens` and
`_s_rg_component` is implemented (the `_use_exact_window()` dispatch,
default-on; #313).  Subclasses without the oracle fall back to the q-order
windowed `solve_rg` + two-cutoff stability certificate (#312).
`_s_rg_component` remains a soft contract (documented default raises)
until each RG subclass provides it.

**Trace pairing — gauge vs flavour grading (#327).** The trace / inner
product is the full FS pairing `Tr_aux(ρ_IR(S_RG)·RG(a)·S_RG) = Σ_{γ′,γ}
⟨(RG_a·S_RG)_{γ′}, (RG_b·S_RG)_γ⟩_aux` over **all** charge pairs.  A same-charge
shortcut (only `γ′=γ`) is a **gauge-grading-only** optimisation: when `Γ_RG` is
a gauge charge the trace legitimately projects to charge-0 (gauge integrated
out), so `γ′=γ` is exact *and* keeps each piece on bounded chords.  When `Γ_RG`
is a **flavour** charge it is *wrong* — `ρ_IR` negates `μ`, so `γ′=γ` forces net
flavour 0, integrating `μ` out *like a gauge charge* and returning the
`μ`-**neutral** series instead of the flavour-**refined** index (it dropped the
`μ+1+μ⁻¹` SU(2) current of `[A_1,A_3]` at `q²` until #327).  Cross-pair all
`(γ′,γ)` and keep the `μ^{γ−γ′}` character; grading the evaluation still bounds
the chord products.

A flavour-refined trace also needs an **internal q-order margin** — the refined
index at `q^K` draws on `S_RG` levels *above* `K` (the structure constants carry
negative-`q` shifts), so compute at `q^{K+margin}` and truncate back (the same
FS-object windowing point as `rg_times_s_rg`).  This is a *windowing artifact,
not an axiom violation*: `ρ_UV` genuinely shifts `μ` (`Γ_RG` is the **IR**
grading, not UV), consistent with `ρ²`-cyclicity — and a charged "residual" that
**marches to the `q^K` boundary as the margin grows** is the signature of
windowing, so grow the margin before concluding an axiom fails.

## Factored / partial RG flows (`then` / `factor_through`, Plan 21)

An RG flow can be split through an intermediate ("mesoscopic") scale
`UV → MS → IR`.  The two `RGKAlgebra` operations are inverse:

* **Combine** — `uv_ms.then(ms_ir)` (`ComposedRG`):
  `RG^UV_IR = RG^MS_IR ∘ RG^UV_MS` and
  `S^UV_IR = RG^MS_IR(S^UV_MS) · S^MS_IR`.
* **Factor / extract** — `uv_ir.factor_through(ms_ir)` (`ExtractedRG`):
  recover `UV → MS` from a known `UV → IR` and a given `MS → IR` over a
  **lattice-compatible** auxiliary.  `S^UV_MS` is solved from the
  factorization above (invert `S^MS_IR`, multiply by `S^UV_IR`, pull back
  through `RG^MS_IR`, then truncate to q-order ≤ K — *compute exactly,
  truncate last*), and `RG^UV_MS = RG^MS_IR⁻¹ ∘ RG^UV_IR` **is the embedding
  `A^UV ↪ A^MS`** (Goal 1.4).

The grading lattices form a short exact sequence

    0 → Γ^MS_IR → Γ^UV_IR → Γ^UV_MS → 0          (Γ^UV_MS = Γ^UV_IR / Γ^MS_IR)

— combine glues `Γ^UV_IR = Γ^UV_MS ⊕ Γ^MS_IR`; extract grades the recovered
flow by the **quotient** `Γ^UV_MS` (`deg^UV_IR` with the `Γ^MS_IR`
coordinates dropped).  These pointed cones are what make every peel
terminate.

The BPS driving class is `rg_flow.factor_through_subquiver(A, subquiver)`:
when `A`'s spec factors as `(factors with an outside-subquiver charge)·
(subquiver spec)` — the subquiver-only entries a contiguous right tail —
`MS → IR` is the subquiver theory flowing to the shared quantum torus, and
the recovered `UV → MS` is the embedding of `A` into that subquiver theory.
The BPS node-deletion flows `SingleNodeRG` / `SubquiverRG` are the special
case; the general factorization lives at the `RGKAlgebra` level (kept
separate from them).

## Directional node-drop flows (`directional_subquiver_rg.py`, Plan 20 T6)

`DirectionalSubquiverRG` / `DirectionalSingleNodeRG` are **the**
directional presentation of a BPS node-deletion RG flow: supply only the
flow data — `auxiliary()` = the IR `BPSKAlgebra` on the surviving nodes,
`grading()` = **total** dropped-node multiplicities (off-cone labels
score negative charges; off-cone-ness is `grading().in_cone`) with cone
generators, and the exact `_s_rg_component` / height-windowed
`rg_generator` — and inherit the complete generic K-algebra API
(apex-peel `multiply`, tRG-mirror `rho`/`rho_inverse`, transported
`trace` with a min-q-order prefilter).  **No UV object exists in the
derived path**, so the identity-on-labels `KAlgebraIso` against an
independently built UV `BPSKAlgebra` is a genuine cross-presentation
certificate — including ρ- and trace-equivariance.

**`S_RG` three ways** (design session 2026-06-10): **(a)** spec already
`(stuff)(IR)` → closed form: the Nahm kets of the spec (a closed form in
the constructor data) peeled against IR F-elements, chart-free;
**(b)** `arrange=True` → BFS over S-preserving local moves (commute
swaps + pentagon collapse/expand) to a stuff-first chamber — labels
never move, so the chamber-calibration trap is structurally absent;
**(c)** `require_stuff_first=False` (the same ket-peel on the spec as
given — the order-by-order factorization; *whether it always yields a
valid flow is open* — empirically robust on every probe so far,
including middle drops; the certificate is the gate) or an
`s_rg_oracle=` injection for hand-factored spectra.  An `f_oracle=`
(e.g. `uv_f_oracle(uv, aux)` — the UV F-peel) accelerates `RG` at
parity; the ρ⁻¹ mirror cross-checks oracle images, so a wrong oracle
fails loudly.

**Harnesses.**  `certify_directional_vs_bps` — the staged certificate
(flow coherence *before* any iso interpretation, then the
identity-on-labels iso battery).  `verify_axioms` — the executable RG
axiomatics (bar / ρ-automorphism / RG-unital/-multiplicative/
-bar-invariant / tRG-intertwining; with `trace_K` also ρ²-twisted
cyclicity + orthonormality).
`scripts/certify_directional_on_dictionary.py` mass-produces and
certifies directional flows over the BPS dictionary (every entry × every
single-node drop) — the "countless RGKAlgebra examples" engine.
`scripts/bench_directional_vs_bps.py` records the speed relation: the
RG-solve-over-smaller-quiver composition is **not** a speed win over
direct UV F-solving (linear A_n family: ~3× on generic labels, falling
with n; 10³–10⁵× slower on deep dropped-cone labels where UV F is
closed-form-trivial); its value is UV-freeness, certification
independence, and uniformity.

**Relation to the other RG classes.**
- `rg_flow.SubquiverRG` / `SingleNodeRG` — **since 2026-06-10, thin
  constructors over the directional class** (previously UV-wrapping
  data-extractors whose KAlgebra ops delegated to the UV).  Same
  constructor `(A_uv, node_indices)`; the UV is retained only as
  `starting_algebra()` (composition endpoints) and as the default
  F-oracle.  `rg_generator` values are unchanged (validated
  term-by-term against the old extraction).
- `ComposedRG` (`then`) — sequential drops compose:
  `D2 = DirectionalSubquiverRG.from_uv(D1.auxiliary(), …)` makes the
  endpoints identity-match, and `D1.then(D2)` ≡ the direct multi-node
  drop (RG images, squashed `S^UV_IR = RG²(S¹)·S²`, structure
  constants — tested).
- `ExtractedRG` / `factor_through` — the general factorization;
  `factor_through_subquiver` consumes the directional `SubquiverRG` as
  its authoritative `uv_ms_flow` (only `RG`/`rg_generator`/
  `_s_rg_component` — parity or better).
- `IsoComposedRGKAlgebra` — post-composing with a `KAlgebraIso` of the
  IR turns a directional flow into a standalone-wrapping one (e.g. the
  tensor auxiliary `A1A2kKAlg(k−1) ⊗ QT(Z₂)` of the A₂ₖ → A₂ₖ₋₂
  two-node drop; the reconstruction-theorem hypotheses are checked by
  `rgkalgebra_iso_via_ir.verify_ir_iso_hypotheses` —
  `tests/test_a1a2k_honest_chamber.py`).
- Per-theory instances: `a1a2k_rgkalgebra` is in substance the
  two-node-drop directional flow with the chord⊗QT auxiliary
  presentation; the empty-survivor corner (all nodes dropped, bare-QT
  IR — e.g. the pentagon at k=1) is `_PentagonViaQT`
  (`tests/test_rgkalgebra_graded.py`), not yet in the directional
  class's BPS-auxiliary scope.
- `single_node_rgkalgebra.py` (root) — the pre-T3 directional first
  cut; superseded, kept for existing callers.

## AbeKAlgebra — the abelianized-presentation contract (Plan 30)

`AbeKAlgebra(KAlgebra)` (`abe_kalgebra.py`, repo root) is the third
presentation tier of the contract family, on par with `RGKAlgebra`
(presented by an RG flow over a graded auxiliary) and `ConeKAlgebra`
(concise cone presentations): a `KAlgebra` **presented faithfully on the
enriched rational quantum torus** — the f-presentation

    x  =  Σ_{m⃗}  f_{m⃗}(𝖖^{m⃗} v) · U_{m⃗},      f_{m⃗} = Σ_{k⃗} f^{(k⃗)}_{m⃗}·μ^{k⃗},

on the (product) cocharacter lattice, with lower-Kapustin labels per
node, ρ the √(full measure) G-cocycle, the trace the (product)
Schur-measure residue with the matter M-factors, and bar the
componentwise `𝖖 ↦ 𝖖⁻¹` (bar-invariance ⟺ W1 palindromicity of every
residual).  **No DOp appears in any tier signature** (ruling D4): the
chart-operator machinery is engine-internal only; the pioneer
DOp-speaking tier is frozen verbatim at `legacy/abe_kalgebra_pioneer.py`.
The literature's abelianization (BFN et al.) is related and
**non-load-bearing** (ruling D2; CLAUDE.md "false friends").

### The contract triple

A realisation supplies exactly three primitives (ruling D3 — minimal):

| primitive | meaning |
|---|---|
| `torus_shape()` | the `TorusShape` of the enriched torus — per-node `RootDatum`s, matter multiplicities, and links (ruling D6, re-ruled 2026-07-02 to the root-datum form; type-A chains via `TorusShape.from_ranks_nf(ranks, Nf)`).  Everything geometric (lattice rank, Weyl blocks, dressing cells, measure) derives from it. |
| `chart(label)` | `L_label` as a `QuiverURQTorus` element — the faithful f-presentation image. |
| `decompose(x)` | a torus element as `Element({label: C(q)})` — the **no-target**, level-ascending canonical read.  Must honest-fail (raise) off its certified scope; never guess, never solve. |

Everything else is **derived at the tier**: `multiply =
decompose(chart·chart)` (the exact torus cocycle product, then the
read); `trace`/`inner_product` (chart-side: the product-measure
residue with matter M-factors; the §6b pairing — no decompose on the
pairing path); `verify_chart_bar` (W1) and `certify_canonical`
(W1 + W2: the q-extreme slice is a single leading Weyl orbit of
multiplicity 1 — the executable Kazhdan–Lusztig acceptance).

**`rho`/`rho_inverse` — the explicit label-level closed form (promoted to
primary, user ruling 2026-08-23).**  On Weyl orbits of joint pairs
`[(m, e)]` (no chamber assumed; both sums Weyl-covariant, so well-defined
mod W):

    ρ^{±1}[(m, e)] = [( −m,  −e + Σ_{α∈Φ: ±⟨α,m⟩>0} |⟨α,m⟩|·α
                              − Σ_{w∈wt(N): ±⟨w,m⟩>0} |⟨w,m⟩|·w )]

— ρ uses the roots / matter weights *positive* on `m`, ρ⁻¹ the negative
half; vector multiplet with `+`, hypermultiplet with `−` (the
weight-valued analogue of the scalar monopole-dimension combination);
flavour irreps star as `w_i ↦ w_i^⋆ ⊗ det_i^{−D_i}`,
`D_i^{±} = Σ_{w∈wt(N_i)} max(0, ±⟨w,m⟩)`.  At `N = Adj`
(`wt(Adj) = Φ ∪ {0}`) the two sums cancel identically and ρ is the
antipode `[(m,e)] ↦ [(−m,−e)]` with `ρ² = id` on the gauge charges; at a
minuscule `ω_k` of U(N) the gauge sum is `pure_un_kalgebra.witten_shift`,
whose `rho_label` maps are this formula's type-A ancestor.  Engine:
`wrq_torus.rho_label` / `rho_level_star` (derivation in its docstring —
the twist factor is a pure monomial at an anti-dominant source atom, and
W2 reads labels off leading data); wired per realisation, since each
realisation owns its label frame (the pure-U(N) family composes with its
lower-Kapustin frame offset).  The former derived route — the torus
G-twist read back through `decompose` — is **demoted to the verifier**
`verify_rho_via_twist` ("chart/twist/decompose then becomes a test",
user, same day); it also remains the fallback for realisations not yet
wired (today: the quiver chains).  Certification:
`tests/test_abe_rho_label.py`; derivation-era battery:
`experiments/rho_label_closed_form.py` [retired 2026-09-19 with the type-A keystone; the measurement stands as recorded].
**The constructive-build rule is contract-enforced**: the tier exposes no solve
entry point, `decompose` is a read, and acceptance is `well_formed`
equality with the intended label.

**`inner_product` — the Schur pairing as a sum over sectors of contour integrals
(user direction, 2026-09-18).**  This is the route the production pure classes
take — `PureGAbeKAlgebra` and `PureSU2KAlgebra` (and, until its 2026-09-19
retirement, `PureSUNKAlgebra`) call
`WRQTorus.inner_by_sector` in their own `inner_product` overrides, asserted
with call counters in `tests/test_kalgebra_rho_star.py` leg 4 (a review found
the first version unreachable from them; fixed the same day) — and the route
`AbeKAlgebra.inner_product` takes for any bare-`WRQTorus` chart.  Not a
refactor of `Tr(ρ(a)·b)` but a separate expression, coinciding with it by the
total-charge-0 part of the cocycle convolution (a theorem, certified rather than
assumed — `tests/test_wrq_sector_pairing.py`, 72/72 at pure SU(2), 32/32 at pure
SU(3), both `w_cutoff` modes).  In the variable `u = 𝖖^{−m}v`,

    I_{a,b} = (𝖖²;𝖖²)_∞^{2·dim}/|W| · Σ_m [u⁰]( M_m(u) · f^a_m(1/u) · f^b_m(u) ),

    M_m = T_{+m}μ · B_m,   μ = ∏_{α∈Φ}(u^α;𝖖²)_∞(𝖖²u^α;𝖖²)_∞,   B_m = T_{+m}(w_m),

`B_m` an exact rational function (`wrq_torus.sector_measure`) and `w_m` the
`v`-frame factor `T_{−m}(G̃_m)·R̃_{−m,m}` (`sector_weight`).  Three facts, each
measured, make this the principled form:

* **The half-shift is the frame, not a convention.**  The two faces
  `Tr(ρ(a)·b)` and `Tr(b·ρ⁻¹(a))` assemble at `T_{∓m}`, so the shift relating
  them is `2m` and the pairing sits at exactly half of it — the midpoint, where
  neither slot is privileged (face1 == face2 == the shipped `b·ρ⁻¹(a)` route,
  16/16 and 9/9).
* **Axiom 5 (ρ-equivariance of the trace) is manifest.**  The two residuals
  enter at `1/u` and `u`, so exchanging `a ↔ b` is `u → 1/u`, under which the
  constant term is invariant and `M_m` is invariant by the exact certificate
  `v̄(B_m) = Q_m·B_m`, `Q_m = T_{+m}μ/T_{−m}μ` a finite product
  (`verify_sector_measure_inversion_symmetric`, pinned 16/16 at
  su_2/su_3/sp_4/g_2 incl. odd `⟨m,α⟩`, non-vacuously — `B_m` itself is not
  inversion-invariant).  **Since 2026-09-19 the certificate is a theorem, from
  the closed form of the measure for every `RootDatum`**
  (`wrq_torus.sector_measure_closed_form`, derived in its docstring and
  certified against the derived object 367/367 at u_2, u_3, su_2, su_3, sp_4,
  so_3, so_5, Spin(5), g_2, U(1)², U(2)×U(1), odd height and non-dominant `m`
  included — `tests/test_sector_measure_closed_form.py`):

      B_m(u) = u^{Σ_{α>0}⟨m,α⟩α} · ∏_{α>0} b_{|⟨m,α⟩|}(u^α),
      b_t(z) = z^t / [(1−𝖖^t z)(1−𝖖^{−t}z) ∏_{j=1}^{t−1}(1−𝖖^{t−2j}z)²],

  the two dressing ladders of the √-measure twist `G_m` cancelling identically
  (the `w₀`-transported `ψ_{−m}⁻¹` carries the same ladder as `v̄(ψ_m)`), the
  scalar in front being `1·𝖖⁰` for every shipped phase convention — at odd
  height because ψ's parity-corrected phase and the ρ block's honest one enter
  with opposite signs — and the half-shifted cocycle factorising over the
  roots.  Each `b_t` is inversion-invariant (symmetric exponent multiset) and
  `Q_m` collapses to the monomial `u^{−2Σ_{α>0}⟨m,α⟩α}`, so `v̄(B_m) = Q_m B_m`
  in one line.  The passage from the `v`-frame, where the code
  computes, to this `u`-frame is a **contour shift** across the annulus where
  the residuals' poles sit — the "contour shift and non-trivial pole
  cancellations" of the user's own description — and is measured, not argued:
  per sector 128/128 at pure SU(2), even `⟨m,α⟩` (review pass); odd `⟨m,α⟩`
  is unmeasured in the `u`-frame.  With flavour, `u → 1/u` inverts `μ` too,
  which is `⋆`: on the matter substrate (`MatterWRQTorus`, since 2026-09-19 on
  the user's requirement) the per-sector measure is `matter_sector_weight` —
  the pure `w_m` times the matter factor of the cocycle `CC[N]_{−m,m}`,
  distributed over the μ-level shifts `r⃗`, with ρ's level star `−k⃗ − D(−m)`
  on the first slot — so exchanging the slots is `v → 1/v` together with the
  level negation, and `I_{b,a} = ⋆(I_{a,b})` is manifest there as well
  (measured 36/36 at SU(2)+1, 25/25 at U(2)+1 and U(2)+2, sector == chart-side
  on the same pairs; `tests/test_matter_sector_pairing.py`).  **The matter
  measure has its closed form too, for every `(G, N)`**
  (`matter_wrq_torus.matter_sector_measure_closed_form`, the `u`-frame object
  `B^N_m = Σ_r⃗ T_{+m}(w_{m,r⃗}) μ^{r⃗−D(−m)}` with the level star folded in):

      B^N_m(u,μ) = B_m(u) · ∏_i ∏_{w∈wt(N_i)} Ξ_{⟨m,w⟩}(μ_i u^w),
      Ξ_c(x) = x^{−max(c,0)} ∏_{j<|c|}(1 + 𝖖^{|c|−1−2j} x),

  the half shift making the matter cocycle's exponents the symmetric set
  `{1−|c|, …, |c|−1}` and the Z-top monomial with the level star supplying
  `x^{−c}` at `c > 0`; `Ξ_c(1/x) = x^c Ξ_c(x)` for either sign of `c`, and the
  hypermultiplet Schur factor `1/(−𝖖x;𝖖²)_∞(−𝖖/x;𝖖²)_∞` (the trace's Nahm
  sum) has shift ratio exactly `x^{⟨m,w⟩}` per weight, so
  `v̄⋆(B^N_m) = Q_m·∏_{i,w}(μ_i u^w)^{⟨m,w⟩}·B^N_m` is the exact matter
  certificate (`verify_matter_sector_measure_inversion_symmetric`, 203/203 over
  the thirteen roster presets, 46 with a non-zero μ-level shift; closed ==
  derived 203/203).  What remains measured rather than derived, on both tiers,
  is the constant term's invariance under the contour move `ι = v̄∘T_{2m}` —
  the `v`-frame face of `u → 1/u`.  Both closed forms are, exactly, the
  `|⟨m,α⟩|`- and `|⟨m,w⟩|`-shifted pairing measure of the user's `K_𝖖-algebras`
  draft (eq. Iexplicit): `B_m` and the `Ξ_c` are the finite ratios converting
  the signed shifts of `T_{+m}(μ_Gμ_N)` into absolute-value shifts (pinned per
  root pair and per weight, test leg 6; Plan 24 D53).
* **Orthonormality is automatic from the seeds.**  At `𝖖 → 0` the lowest-order
  term of `M_m` is the **m-Levi Vandermonde** `∏_{⟨m,α⟩=0}(1−u^α)` exactly (no
  monomial, sign `+`; 12/12 at su_2/su_3/sp_4), so the leading pairing is Schur
  orthogonality of the seed Levi characters against the Levi Weyl measure and
  `I_{a,b} = δ_{a,b} + O(𝖖)`.

Computationally the sum touches only the shared magnetic support and never
forms `ρ(a)` — `WRQTorus.pairing_residual` / `inner_by_sector`; ~2× faster than
the convolution route at SU(2) with supports to 9 — and the per-sector pieces
are summed **before** the residue is taken, because `trace_residual` is
non-linear at `w_cutoff=True` (the test pins that this guard is load-bearing).
Charts with μ-levels on `MatterWRQTorus` take the same route
(`MatterWRQTorus.pairing_residual` / `inner_by_sector`, the μ-refined residue
applied to the per-level rows, Nahm window tied to `K`); `QuiverWRQTorus` has
no live producer since the 2026-09-19 retirement of `UNQuiverKAlgebra`.  `aux_space.py`'s
vacuum-state pairing (Plan 40, the user-ruled primary *definition* of the
Schur pairing, CLAUDE.md) is the same pairing written on the auxiliary
space: `AuxSpace.from_datum(su_2).schur_pairing == ` this route, pinned 16/16
at `K=6` (`tests/test_kalgebra_rho_star.py` leg 4).  Its `w_m` (the weight
*including* the shifted Schur measure) is this paragraph's `M_m` up to
normalisation; the letter `w_m` here (`sector_weight`) is the measure-free
`v`-frame factor — two normalisations of one object, both documented.  Which
of the two is the tier's *stated* definition is a contract-surface ruling
for the user; nothing here retires Plan 40.

### The substrate

> **Superseded by rulings D9/D10 (2026-06-30 / 2026-07-01) — read this box
> first.**  The contract-facing substrate is now the **group-general WRQTorus**
> family (`wrq_torus.WRQTorus` / `matter_wrq_torus.MatterWRQTorus` /
> `quiver_wrq_torus.QuiverWRQTorus`), *selected by* `torus_shape()` and exposed as
> the `torus()` property (`abe_kalgebra.AbeTorus` does the selection: `pure` /
> `matter` / `quiver` by node count and matter content).  There is **no per-tier
> substrate class**.  The `URQTorus` family described below stays as the
> substrate of `aux_space`'s URQ form and of `kalgebra_samples.Sqed1KAlg`
> (its Route-A role — `PureUNKAlgebra.chart()` returned the `vr_to_tr`
> transport of its URQ engine image — ended with that keystone's retirement on
> 2026-09-19, when the chart-level helpers moved into `urq_torus.py`), so
> nothing is lost, but a *new* realisation should declare a
> `TorusShape` and let `torus()` pick the WRQ type.

`quiver_urq_torus.QuiverURQTorus` is the cell-table core (atoms `U_{m⃗}`
on the product cocharacter lattice; per-link rungs on negative
pair-differences `d_{jl} = m⁽ᵃ⁾_j − m⁽ᵃ⁺¹⁾_l`, per-node-flavour rungs on
negative `m_j`; the gauge cocycle `CC` × finite net-numerator matter
cocycles; block GTwist ρ/ρ⁻¹ with the q-free Z-top monomial division and
the level star `k_slot ↦ −k_slot − D_slot`; product Schur-measure trace;
joint per-node-Levi recognition with explicit dominance tie-breaks).
`URQTorus` is the per-node pure substrate the core imports;
`MatterURQTorus` (U(N)+N_f) is the exact 1-node view (delegated), and
keeps the native constructive-build-rule **build engine** (product-and-peel)
matter-side.

### The product law with matter — the cocycle for `U_m`, and the two residuals

*Recorded here because it was previously stated only in module docstrings
(`matter_wrq_torus.py`, `matter_star_bubbling.py`) and in
`g_matter_abelianized.md`, so the matter tier's product law could not be read
off any root design doc — a documentation gap the user flagged, 2026-07-30.*

On `MatterWRQTorus` a residual carries a **flavour level** per hypermultiplet
slot, `f = {atom m: {k⃗ ∈ Z_{≥0}^{#slots}: TorusRational}}` (one slot per
irreducible matter summand — ruling TM7), and the atoms multiply as

    U_m · U_{m'}  =  CC[N]_{m,m'} · U_{m+m'}               (flavour levels convolved)

**The gauge backbone is *identical* to pure gauge** — which is what makes matter
an additive extension: `CC[N] = CC · (CC[N]/CC[0])`, with `CC = CC[0]` the
pure-gauge cocycle (`wrq_torus.CC`) and `CC[N]` the `(G, N)` one
(`matter_wrq_torus.CC_N`).

    R_{m,m'}  =  T_{m+m'}(CC_{m,m'})                       (the bare-argument form)
    T_p : v^λ ↦ 𝖖^{⟨p,λ⟩} v^λ   (the shift; `TorusLaurent.q_shift(p)`)

Matter does not touch the backbone.

**`CC` is THE PRIMITIVE, defined by a closed form** (user rulings, 2026-08-24:
*"the key is the cocycle in `U_m` products.  Ideally it would be a primitive
object instead of being built from pieces like `ψ`"*; then *"In the definition
of the rational quantum torus for `Σ_m f_m(𝖖^m v)·U_m`, `R̃` is clearly the truly
primitive object"*; then the name, *"`CC` is fine for cocycle.  `CC` for pure
gauge, `CC[N]` combining gauge and matter"*).  `CC` is what the product law is
written in; `R = T_{a+b}(CC)` is derived.

**`CC` is the pure-gauge cocycle and `CC[N]` the `(G, N)` one**, so `CC = CC[0]`
is a genuine specialisation matching the repo's `(G, N)` / `(G, 0)` convention —
and `CC[N]` is exactly what the product law multiplies by.  The matter-only piece
is the ratio `CC[N]/CC[0]` and carries no name of its own.  ⚠ **`N` must be a
representation whose weights are allowed by the global form** (user, 2026-08-24):
at `PSU(N)` the fundamental is not admissible and the adjoint is, so `CC[N]` is
simply not defined for an inadmissible `N` (`matter_wrq_torus.CC_N`,
`verify_matter_weights_admitted`).

Both charged sectors are products over the **charged directions**, and a
direction contributes exactly when the two magnetic charges pair with it in
**opposite signs** (user: *"the (gauge) cocycle receives contributions from roots
for which `m` and `m'` inner products have opposite signs"*, and *"same for the
matter cocycle"*) — measured exhaustively.  Writing `A`, `B` for the two
pairings, both factors live on one **Clebsch–Gordan range**

    d = ||A| − |B||,      D = |A| + |B|,      stepping by 2

and the vector/matter parallel is then exact:

| | spans | as | multiplicity |
|---|---|---|---|
| vector multiplet, over roots `α` | the **closed** range `d … D` | denominators `1/(1 − 𝖖^k v^α)` | ends once, interior twice |
| matter, over weights `w` of `N` | its **strict interior** `d+1 … D−1` | numerators `(1 + 𝖖^{sgn(A)k} μ v^w)` | once each |

Explicitly, for `A > 0 > B` and `z = v^α` (with `A < 0 < B` the bar image):

    cc(A,B) =           𝖖^{|A||B|} · z^{min(|A|,|B|)}
              ────────────────────────────────────────────────────────
              (1 − 𝖖^d z)·∏_{k=d+2,…,D−2}(1 − 𝖖^k z)²·(1 − 𝖖^D z)

Nothing in that mentions `ψ`, the atom phase `S`, the Weyl transport or the D31
phase coboundary: the closed form absorbs them all, odd `⟨Σ⁺,m⟩` included.  Two
corollaries worth having.  **`CC_{a,b} = 1` whenever `a`, `b` and `a+b` share a
closed Weyl chamber** — all the content is chamber-crossing.  And **`CC`
satisfies the bar axiom at torus level**, `bar(CC_{a,b}) = CC_{b,a}`, because
every ingredient above is symmetric in `|A|, |B|` so the swap and the charge
conjugation coincide; `R` does *not* satisfy it (measured: 285/625 at SU(3)
against `CC`'s 625/625), which is a second reason `CC` is the primitive.

The matter piece is **already in this frame** — its own definition carries the
shifts `T_{−m'}`, `T_m` — so `CC[N]` needs no further shift.

**`ψ` and `Z` are demoted to trivializations** whose *"main role … is to allow
the formulation of the star axiom"* (user).  `δψ = R` and `δZ = W` in the twisted
sense; since `R` and `W` are now defined independently, those equalities are
**emergent** evidence rather than tautologies (`verify_psi_trivializes_cocycle`,
`verify_Z_trivializes_matter_cocycle`), and so are the twisted cocycle condition
and Weyl covariance.

Two `ψ`s must be kept apart.  The **honest** `ψ` trivializes `R` on the nose but
carries a fractional power of `(−𝖖)` at odd `⟨Σ⁺,m⟩`, so it does not live in
`Z[𝖖^{±1}]`.  The **floored** `ψ` — *"`ψ` with a fractional power of `(−𝖖)`
stripped off"* (user) — is what `dressing_psi` returns; it therefore **does not
trivialize `R`**, but the defect is a `v`-free scalar `(−𝖖)^{δ(S_honest−S_used)}`,
*"which does not affect the star axiom because it is an overall factor"* (user)
— (★) being a homogeneous residue-cancellation condition.  So the floored `ψ` is
safe exactly where `ψ` is still used, and keeps every surface in integral powers
of `𝖖`.

**Why the fraction is in `ψ` and not in `Z`, despite their similarity** (user's
question, 2026-08-24).  `ψ` is a *square root* and `Z` is not.  `ψ` normalises
the atom, and the atom enters the Schur pairing **twice**, so its normalisation
is pinned by a *bilinear* condition (`I = δ + O(𝖖)`) — which fixes `c²`, hence
`c` only up to a root: `S = −⟨ρ,m⟩ = −½⟨Σ⁺,m⟩`, half-integral exactly at odd
height (`_psi_monomial_data`: *"Orthonormality PINS `S = −⟨ρ,m⟩`"*).  `Z` is not
a normalisation of anything — it is the matter one-loop numerator attached to the
atom, entering **linearly**, so no root is taken.  The structural shadow is in
the ladders: matter's exponents `{c+1, c+3, …, −c−1}` are symmetric about `0` for
every charge (the `+1` does it), so `Z` needs no normalising monomial, while the
gauge exponents `{0, 2, …, 2A−2}` are centred at `A−1` and so `ψ` does.

Battery: `tests/test_primitive_cocycle.py`.

⚠ **The shift used to be written `S_m` here as well as `T_p`, as if they were
two operators.**  They are not: both are `TorusLaurent.q_shift`, and
`S_m = T_{2m}` identically.  The `S_` spelling is **retired** (2026-08-25):
`S` already carries three meanings — the spectrum generator `S`, the flow's
`S_RG`, and the atom phase `S(m)`, which appears in the very display above as
`S_honest`/`S_used` — so an `S_`-subscripted shift is *"a collision waiting to
happen"* (user, 2026-08-25; the duplicate spelling was noticed the day before,
*"I thought the shift was referred to as `T_m`"*).  `T_p` was already the
spelling in the matter cocycle `W_{m,m'} = T_{−m'}(Z_m)·T_m(Z_{m'})/Z_{m+m'}`,
so this removes an alias rather than adding a symbol.  Sites updated:
`wrq_torus.py`, `urq_torus.py`, `experiments/neumann_wavefunction_actions.py` [retired 2026-09-19 with the type-A keystone; the measurement stands as recorded],
this file.  **Not** updated: the dated record in
`restructuring_plans/24_nonminuscule_groups/decisions.md` (history is left as
written), and the `export/` package copies, which are under
`export/README.md`'s sync-direction warning and must receive the hunk rather
than be edited in place.

**Matter enters through one extra factor**, built from the *matter rung
dressing* at a charge `n`

    Z(n)  =  ∏_i ∏_{w ∈ wt(N_i), c = ⟨n,w⟩ < 0} ∏_{s=0}^{|c|−1} (1 + 𝖖^{2s+c+1} μ_i v^w)

— the ladder of **matter zero modes** (`μ_i` = slot `i`'s flavour fugacity), i.e.
the matter one-loop numerator at charge `n`, bar-centred so the exponents
`2s+c+1` run symmetrically from `c+1` to `−c−1`.  It is polynomial in `μ` and
`v`: **`Z` introduces no new poles**, so denominators stay gauge roots.  Then

    W_{m,m'}  =  T_{−m'}(Z_m) · T_m(Z_{m'}) / Z_{m+m'}      per flavour level

a **finite net numerator**, computed once by triangular division in `μ` with a
rung budget that honest-fails rather than truncating
(`matter_wrq_torus._matter_cocycle`).  `ρ` is likewise the pure WRQ twist
followed by the image atom's matter factor (divide by the 𝖖-free `Z`-top
monomial `v^{E(−m)}`; level star `k_i ↦ −k_i − D_i(−m)`, `D` = the slot's rung
count).

**Two residuals, and which one is the chart — because the two theories have
different atoms.**  This is the point that makes the rest of this section follow
rather than having to be remembered (user, 2026-07-30): comparing a `(G, N)`
canonical against an `RGKAlgebra` built over the `(G, 0)` `AbeKAlgebra` means
converting between the two theories' atoms, and

    U^{G,N}_m  =  Z_m · U^{G,0}_m                (the matter factor, per atom)

So an element written on the **matter** atoms with residual `Q` is the *same*
element written on the **pure** atoms with residual `F = Z·Q`.  That is the whole
content of the two presentations below — not a normalisation convention but a change
of atom basis — and **the matter factor is reproduced by the `RGKAlgebra` RG-solver**,
so `Z·chart == rg_chart` is a genuine cross-check of that factor rather than a
comparison with `Z` put in by hand on both sides.

| | residual | who holds it |
|---|---|---|
| **dressed** | `F` | the RG image `RG(L_{m,e})` of the flow (`GMatterOverPure.rg_chart`) |
| **un-dressed** | `Q = F/Z` | the **chart** stored by `GNAbeKAlgebra` |

`F = Z·Q` is the relation, and presenting `F` as the chart would count the
dressing **twice** (ruling: "the chart is `Q`, not `F`").  Consequences worth
keeping in view: the divisibility of `F` by `Z` per cell is a genuine condition
(`matter_star_bubbling.divide_by_Z` / `matter_multislot.divide_by_Z_vec`), and
`Q[0⃗] = pure` (ruling D3) is what makes `decompose` a *read* — the lowest
flavour-level slice is a combination of pure-`G` charts, so it has no target and
cannot fabricate.

**And on the leading Weyl orbit the residual is matter-independent** — user ruling,
2026-07-30: *"the seed written in terms of `U_m` which include matter contributions
to the `R` cocycle is matter-independent."*  Because the atoms carry the matter
factor `W` above, writing the leading orbit in the `U_m` basis leaves nothing
matter-dependent in it: its un-dressed residual is *literally* the pure canonical's,
so `Q[k⃗]|orbit = 0` for every `k⃗ ≠ 0⃗`.  This is why the `(G, N)` constructor
*writes* the leading orbit in rather than solving for it (the extremal cells carry no
unknowns) — there is no freedom there, and it is a statement about the presentation
rather than a measured degree law.  Plan 24 D35 addendum.

### Generalized global forms — the line lattice and its Weyl quotient (2026-08-24)

The label space of `L_{m,e}` is the **Weyl quotient of a lattice `Λ`** that sits
between `Q^∨ × Q` and `P^∨ × P` and on which the **Dirac pairing is integral**
(user, 2026-08-24).  The traditional global forms are the *product* case
`(coweights of G) × (weights of G)`; the general case also covers the correlated
lattices — a discrete theta angle — and at `su(2)` gives the three forms SU(2),
standard SO(3), and *"that which requires odd `e` weights when `m` weight is
odd"* (user), whose only extra line is the dyon `(ω^∨, ω)`.

Because `Λ ⊇ Q^∨ × Q`, it is the preimage of a **finite** subgroup
`L = Λ/(Q^∨ × Q) ⊆ (P^∨/Q^∨) × (P/Q)`, and the Dirac pairing descends to the
linking pairing there.  So *admissible ⟺ `L` isotropic*, *maximal ⟺ `L`
Lagrangian* — and **maximality is not required** (user ruling): a non-maximal
isotropic `L` is a consistent, merely incomplete, set of lines.  `W` acts
trivially on both quotients, so **every** admissible `Λ` is automatically
`W`-stable and the existing dominant-representative frame is already a section of
`W\Λ`: the Weyl quotient costs nothing.

Carried by `global_form.LineLattice` — the same class and the same name as
before, now taking either `H=` (a centre subgroup, the product forms) or
`classes=` (generators of an arbitrary class-pair subgroup).  Class labelling is
**derived from Smith normal form** rather than from a searched generator, so
non-cyclic centres (`Spin(4k)`'s `Z₂ × Z₂`) work; the two labellings are dual by
the theorem `C^{-1} = V D^{-1} U`.  ⚠ `admits` is **no longer**
`mag_admits and elec_admits` — that conjunction is right only for a product form;
the two side predicates are now the *fibres* of `admits` over `e = 0` and `m = 0`
(unchanged on every product form).  The magnetic side tests a **coset**
(`m ∈ M_0 + lift(k)`) and the electric side a **class**, the asymmetry being the
repo's storage convention — cocharacters in the coroot basis, weights in the
fundamental-weight basis — which is also what keeps `U(N)` (non-semisimple, its
lattice meeting every class) working unchanged.

`line_lattice_torus.py` (repo root) carries the two exercises the user set
before the rational variants.  `conventional_quantum_torus(lines)` takes a
`Z`-basis of `Λ`, reads the Dirac matrix on it, and hands it to the existing
`QuantumTorusKAlg` — integral exactly when the lattice is admissible, so an
inadmissible lattice is refused by arithmetic rather than by a separate check.

Two sibling tori live there, distinguished by the user's own letters (`f` on the
dressed atoms `U_m`, `d` on the plain generators `u^m`).  `DRationalTorus(lines)`
(name the user's, 2026-08-24) represents `Σ_m d_m(𝖖^m v)·u^m`: no dressing `ψ`
appears, so the question of whether the atom is `ψ_m(v)·u^m` or `u^m·ψ_m(v)` does
not arise — the demonstrative step.  `FRationalTorus(lines)` represents
`Σ_m f_m(𝖖^m v)·U_m` on the **dressed atoms**, which is the torus the tier
ultimately wants.

**The only difference between them is the cocycle.**  `U_a` commutes past a
function of `v` exactly as `u^a` does — the dressing is itself a function of `v`
— so the shifts are identical and the whole content of the dressing enters
through `U_a U_b = R_{a,b}(v)·U_{a+b}`:

    [fg]_n(v) = Σ_a f_a(𝖖^{a−n} v)·f'_{n−a}(𝖖^{a} v)·CC_{a,n−a}(v)

with `CC_{a,b}` (`wrq_torus.CC`).  Since `CC` is the **primitive** closed form,
this exercises it directly: associativity of this
product *is* the twisted 2-cocycle condition.  `CC` carries only root weights, so
it drops into the residual's rational factor without disturbing the storage
discipline, and its `𝖖`-powers are integral.  Cross-checked against
`wrq_torus.WRQTorus`, whose own `__mul__` is this same law — an independent
oracle rather than a restatement.  The two classes share a private base because
they share a *representation*; they are different algebras, not a subtype pair.

Only the **bare** residuals `d_m(v)` are stored (user: *"using `𝖖^n v` can be
dangerous"*), as `{m: {e: rat}}` meaning `d_m(v) = Σ_e v^e·(a rational
function of the root characters `v^α`)` with `(m, e)` a line of the lattice and
the second factor a `TorusRational` carrying **root** weights only.  That factor
is deliberately left unnamed rather than written `R_{m,e}`: `R_{m,m'}` is this
tier's settled notation for the atom cocycle `U_a U_b = R_{a,b}(v)·U_{a+b}` (a
few sections above), and a two-index `R` for a different object would collide.  That split is load-bearing: a shift sends `v^α ↦ 𝖖^{⟨c,α⟩}v^α` with
`⟨c,α⟩ ∈ Z` always, so the whole fractional content sits in one scalar per
residual, and in the product law the two such scalars combine into the Dirac
pairing — an integer exactly by admissibility.  Without it the frame would not
close over `Z[𝖖^{±1}]` at a correlated form, where `⟨ω^∨, ω⟩ = ½`.  The product
law is the user's,
`[dd']_n(v) = Σ_{n'} d_{n'}(𝖖^{n'−n}v)·d'_{n−n'}(𝖖^{n'}v)`, and on monomials
`d_n(v) = δ_{n,m}v^e` it reproduces `X_{m,e}` of the conventional torus.

Two properties the class enforces rather than assumes.  **Residual storage is
canonical mod the root lattice**: `v^e·R` and `v^{e−nα}·(v^{nα}R)` are the same
residual, so `e` is reduced modulo `Q` and the root monomial absorbed into `R`
(user, 2026-08-24) — otherwise equal elements would compare unequal and sums
would not merge.  Reduction is modulo `Q`, not by electric class, because at a
non-semisimple datum the class does not determine `e` mod `Q`.  And **the algebra
depends on the lattice, not just the datum**: admissibility gates every residual,
so a `DRationalTorus` is the torus of a *global form*, never "the torus of `G`";
terms that do not fit one consistent lattice are refused at construction, and if
that check is disabled the product raises on the resulting fractional `𝖖`-power
rather than returning a wrong answer.

**`G₂` has exactly ONE global form, and the framework says so structurally**
(user question, 2026-08-25).  A line lattice sits between `Q^∨ × Q` and
`P^∨ × P`; at `G₂` the weight lattice **equals** the root lattice, so the two
bounds coincide and there is nothing in between — allowing non-maximal lattices
does not help either, since there is no gap to sit in.  Verified two independent
ways: `centre_labelling(g_2())` returns empty invariant factors (trivial centre),
and the Cartan determinant is `|P/Q| = 1` (against 3 at SU(3), 2 at Spin(5) and
Sp(2)).  `simply_connected_lines` and `adjoint_lines` return *the same* lattice
there — the single class pair `((), ())` — which is the correct answer rather
than a degeneracy of the code.

So `G₂` is the least interesting datum for global forms and among the most
interesting for everything else: non-simply-laced, no minuscule cocharacter, the
richest wall structure.  Where the global-form questions actually live is the
data with non-trivial centre.

Record: Plan 24 decisions, the generalized global forms ruling.  Battery:
`tests/test_line_lattice_torus.py` (45 checks, conceptual / practical).

### The `L_{m,e}` construction — the axioms, and the single route (2026-08-25)

**User direction.**  *"`AbeKAlgebra[G,N]` can become a fully functional
`KAlgebra` (conjecturally) for all `G` and `N`"*, *"with all axioms and the
resulting `L_{m,e}` construction clearly presented and coded in a principled and
transparent manner"*; *"no cones, no minuscules, no product and peel — these are
all obsolete"*; *"all is needed is star + `O(𝖖)` axioms"*, which *"should allow a
direct derivation of `L_{m,e}` for all `m` and `e`"*; and *"even Wilson must
follow from the axioms"*.

So there is now **one route**, at every `G`, every `N`, every label:
`star_bubbling.solve_canonical` (pure) and
`matter_multislot.solve_canonical_matter_vec` (with matter).  It is given the
tier's own presentation data plus five conditions, all stated on the
f-presentation `x = Σ_n f_n(𝖖^n v)·U_n`:

| | what | role |
|---|---|---|
| the presentation | atoms `U_n` on the line lattice's Weyl quotient; products governed by the primitive cocycle `CC` (pure) / `CC[N]` (with matter) | the ring the answer lives in |
| the seed (W2) | the `𝖖`-extreme slice is a single leading Weyl orbit of multiplicity 1, `f_{w·m} = w·χ^{L_m}_e` | fixes the normalization — it is what *names* the label |
| support | `n` ranges over the tropical support `conv(W·m) ∩ (m + Q^∨)` | which cells can be nonzero |
| bar (`W1`) | every residual is `𝖖`-palindromic | |
| `O(𝖖)` | `val_𝖖 f_n ≥ 1` off the leading orbit (with matter: on the quotient `Q = f/Z`, not on `f`) | |
| (★) | the element's `𝖖`-difference operator preserves `Λ = R(G)` **of this form** | the guard that licenses the solve |

**Which of these are selecting conditions, and which are consequences** (user,
2026-08-25: *"the axioms presumably include the seed shape, bar invariance,
`O(𝖖)` bubbling, star.  Am I missing something?"*).  Those four **are** the whole
list.  The other two rows are of a different kind: the presentation is the
ambient ring, presupposed rather than imposed; and the **support bound is
subsumed by (★)** — measured, and with a mechanism.

*The measurement.*  Graft onto a genuine `L_{m,e}` a residual taken from another
canonical's bubbling, at a cell `p` OUTSIDE `conv(W·m)`.  Since bar acts cellwise
(`𝖖 ↦ 𝖖⁻¹`, `v` fixed) such a residual is itself palindromic, and a bubbling
residual has `val_𝖖 ≥ 1` — so the perturbed element satisfies W1, W2 and `O(𝖖)`
**by construction**, and (★) is the only axiom that can see it.  It sees it every
time: **30/30 grafts rejected, 0 accepted**, at SU(2) (`m=(1,)`, `(2,)`, dressed
and undressed), SU(3) and Spin(5), each with the true canonical passing (★) as
the positive control.  Battery: `experiments/support_bound_subsumed.py`.

*The mechanism.*  (★) is a residue cancellation between a cell and its
affine-Weyl reflection partners, so it is the one condition that **couples cells
across magnetic charge**.  A nonzero cell outside `conv(W·m)` demands matching
residues at its partners — including extremal cells, whose denominators R3 pins
exactly to the walls predicted from within `conv(W·m)`.  The support bound is
therefore a consequence of (★) + W2, not an extra assumption.

This is evidence, not proof, and it is the same shape as the precedent: Weyl
covariance also looked like an axiom and was measured subsumed.  (An earlier
probe here — dilating the solver's cell list — was **inconclusive** rather than
negative: `joint_solve`'s `K_d` reads the wall set off the support list, so
enlarging it manufactures spurious walls at the extremal cells and the R3 seed
check fails, 5/5.  That is a harness artifact; the graft construction avoids it
by never touching the solver.)

*The dilation, conclusive* (2026-09-24; Plan 42, `conj:abeKalgebra/support`).
`joint_solve(support=)` removes the artifact: beyond the hull a seed cell meets
partners on walls where it has NO pole, so its walls are a subset of the predicted
ones, and the seed numerator is taken over the full predicted denominator — it
vanishes on the extra walls, which forces the partner's residue there to zero.
Solved on the hull grown by the smallest dominant coroot, (★) + bar + `O(𝖖)` have a
unique solution that vanishes outside `conv(W·m)` and equals the production line:
157/157 pure lines at the battery's extensive depth.  The control removes the
residue rows: the `O(𝖖)` rows are then cell-local, and 208 of the 365 added
Weyl-orbit representatives admit a nonzero bar-invariant `O(𝖖)` filling — on
those it is (★) that empties them; the outermost added orbit has no poles of its
own and is emptied by bar + `O(𝖖)` alone.  With matter (2026-09-26) the same
measurement runs at every μ-sector, through
`matter_multislot.solve_canonical_matter_vec(support=)`, every sector solved; the
battery's support row records the populations and the counts.

The unknowns are the numerator coefficients of the bubbling cells inside a
`v`-window that is a **cutoff, never a selector** — the box certificate re-solves
one `pad` wider and demands the same element.  The system is linear over `Z`;
`solve_canonical` accepts only a **unique** solution and then applies `W1`, (★)
and the box certificate before handing anything back.  Nothing in that list is a
generating set, a cone, or a preferred family of lines.

**The axioms are executable, in one place.**
`star_bubbling.verify_axioms(datum, m, e, x)` runs all five against a
**finished** element and returns `{condition: (ok, detail)}`.  It exists apart
from `solve_canonical`'s internal guard because the guard runs on what the solver
just built, whereas this runs on anything — in particular on the retired routes'
output, so "the constructive build satisfies the axioms" becomes a test rather
than a restatement.  Its `O(𝖖)` check is exact, with the truncation bound
discharged rather than assumed (every denominator factor has non-negative
`𝖖`-valuation, so cutting the inverse-denominator series at
`max(0, −min qdeg num)` cannot drop a `𝖖^{≤0}` term).  What it does **not**
certify is uniqueness: that is a property of the system, not of an element, and
it stays the solver's own first guard.

Each condition was shown to catch something the others do not — the negative
controls, in `tests/test_pure_g_abe_kalgebra.py`: a **bubbling-stripped** SU(2)
`L_{(2,),0}` passes W1, W2, support and `O(𝖖)` and is rejected by **(★) alone**
(the 23/23 discrimination, live); a good element under the **wrong** `e` is
rejected by W2 alone; and `L_1·L_1` asked as `L_1` is rejected by support, W2 and
`O(𝖖)` together.

**Wilson is the degenerate case, not a special case.**  At `m = 0` the tropical
support is the single cell `{0}`, so the system has **no unknowns**: the seed
*is* the element, `L_{0,e} = χ_e(v)·U_0`.  `W1` holds because `χ_e` is `𝖖`-free,
and (★) holds because `R(G)` is a ring — checked, 14/14, and both add zero
constraints there.  Measured 7/7 Wilson labels across SU(2), SU(3), Spin(5),
Sp(2), G₂, U(2).  Two qualifications worth keeping visible, because they are
where the content actually sits:

* `χ_e` enters as the **seed**, i.e. as the W2 normalization — it is not solved
  for (user, 2026-08-25: *"good point, it is a seed"*);
* electric admissibility of `e` at a non-simply-connected form is enforced by
  `lines.admits` inside `chart`, **not** by (★) — `star_bubbling.criterion`
  tests against the datum's own `R(G)`.

A label handed over in a **non-dominant representative** is not a failure of any
of this: labels are `W`-orbits of pairs, so `(0, α₁)` at SU(3) is the adjoint
Wilson line and `fold` transports it to `(0, θ)`.  `chart` folds for you; the raw
solver wants the normalized representative and now says so.

**Retired, not erased.**  The constructive zoo — `wilson`, `minuscule`,
`monoid`, `closed_form`, `twist`, `cone`, `peel` — is intact behind
`PureGAbeKAlgebra(..., constructive_routes=True)`, which restores it ahead of the
solve exactly as it was.  It is kept for the reason the retired
`recursive_spectrum` is kept: it is an *independent* construction of the same
element, which is what makes agreement between the two evidence rather than
tautology.  Measured 33/33 agreement over the label sweep (SU(2), SU(3),
Spin(5), Sp(2), G₂, U(2) — Wilson labels included), at a cost that goes both
ways: the solve is faster at the hard end (G₂ `(1,0)`: 5.3 s vs 7.0 s) and
slower where a closed form existed (SU(3) `m=(2,0)`: 4.6 s vs 0.04 s).  Stage 2
reintroduces the ones worth having as **declared optimizations** — some valid at
all `(G, N)`, some specific to a `G` and an `N` — rather than as a preference
list whose winner is label-dependent and undocumented.

The `(G, N)` tier needed no change of shape: `GNAbeKAlgebra._base_tower` was
already the θ-twist plus the guarded solve.

### The `KAlgebra` axioms from the `AbeKAlgebra` axioms — what is derived, and the one thing that is not (2026-08-25)

**User direction.**  *"Of course some of the KAlgebra axioms follow analytically
from the AbeKAlgebra axioms.  For example, the definition of `I_{a,b}` in
AbeKAlgebra together with the `O(\fq)` bubbling axiom should imply the
`δ_{ab} + O(\fq)` axiom for `I_{ab}`"*, *"and the implication should be an
analytic statement, not just a numeric test"*, *"same for bar etc."*  And the
sharpening that reorders the whole question: *"The fact which is not obvious is
actually that the `L_{m,e}` **form an algebra**"*, so *"testing that the
multiplication works is particularly important"*.

| `KAlgebra` axiom | status | where the content is |
|---|---|---|
| unit `L_{0,0}·L_a = L_a` | **derived** | `chart(L_{0,0}) = χ_0·U_0 = 1`, the torus unit |
| bar antimultiplicativity | **derived** | W1 + `bar(CC_{a,b}) = CC_{b,a}` |
| `ρ` automorphism, `ρ⁻¹ρ = id` | **derived** | `ρ` is conjugation by the √measure — an automorphism of the substrate |
| `ρ²`-twisted trace cyclicity | **substrate** | a property of the measure residue; no canonical-basis input at all |
| orthonormality `I_{a,b} = δ + O(𝖖)` | **derived** | W2 + `O(𝖖)` + two substrate lemmas |
| associativity | **derived, given closure** | the torus product is associative; `chart` is faithful |
| **closure** — `L_a·L_b ∈ span_{Z[𝖖^{±1}]}{L_c}` | **NOT derived** | ⟺ the spanning statement below |

So all but one are formal once the substrate is fixed, and a verifier that passes
them is confirming arithmetic rather than probing the mathematics.  The single
substantive fact is closure, and that is where the testing belongs.

#### Bar antimultiplicativity, derived

`bar` is an **antiautomorphism of the substrate**, and precisely because the
cocycle satisfies the bar axiom `bar(CC_{a,b}) = CC_{b,a}` (the primitive
cocycle ruling; it is `CC` that has this property, *not* the bare-argument `R`).
Hence `bar(x·y) = bar(y)·bar(x)`.  Now let `x = chart(a)`, `y = chart(b)`; by W1
both are bar-invariant, so

    bar( chart(a)·chart(b) )  =  chart(b)·chart(a).

Applying `decompose` and using that it commutes with bar — each `L_c` is
bar-invariant, so bar acts on the coefficients alone as `𝖖 ↦ 𝖖⁻¹` — and that the
`L_c` are linearly independent:

    C^c_{ab}(𝖖⁻¹)  =  C^c_{ba}(𝖖).                                   ∎

No numeric input.  Controls (they check the two premises, not the conclusion):
`bar(CC_{a,b}) = CC_{b,a}` 27/27 and `bar(xy) = bar(y)bar(x)` 29/29 across SU(2),
SU(3), Spin(5).

#### Orthonormality, derived from W2 + `O(𝖖)`

`I_{a,b} = Tr(ρ(L_a)·L_b)` is the magnetic-0 Schur residue of `ρ(x)·y`.  Since
`ρ` reverses the magnetic charge, `(ρ(x)·y)_0 = Σ_n [ρ(x)]_{−n}·y_n·CC_{−n,n}` —
a **sector-diagonal** sum, pairing `x_n` against `y_n` through a weight `w_n`
built from the measure, `CC_{−n,n}` and the `ρ`-cocycle.  Then:

* **Lemma A (the weight costs nothing).**  `val_𝖖 w_n = 0` for every `n`.  The
  Vandermonde's `(−𝖖)^{−|⟨n,α⟩|}` is a *negative* power, and it is cancelled
  exactly by `CC_{−n,n}`.  So no sector can contribute below the `𝖖`-order the
  residuals themselves carry.  *Measured, not proved*: SU(2) `n ≤ 3`, SU(3),
  Spin(5) — valuation 0 in every sector.
* By **W2 + `O(𝖖)`**: `val_𝖖 x_n = 0` exactly on `n ∈ W·m_a`, where
  `x_n = w·χ^{L_{m_a}}_{e_a}`, and `≥ 1` at every other cell; likewise for `y`.
* Hence `I_{a,b}[𝖖⁰]` can only receive contributions from `n ∈ W·m_a ∩ W·m_b`,
  and only from the leading-orbit values there.  Two cases:
  * `m_a ≠ m_b` as **dominant representatives** ⟹ the orbits are disjoint ⟹ no
    sector contributes ⟹ `I_{a,b} = O(𝖖)`;
  * `m_a = m_b = m` ⟹ `I[𝖖⁰]` is the pairing of the two **leading orbits alone**.
* **Lemma B (leading-orbit orthogonality).**  That pairing is `δ_{e_a,e_b}`: the
  orbit sum over `W/W_m` against the `1/|W|`, reducing to orthonormality of the
  Levi characters under the Levi Weyl measure.  *Measured*: 104 pairs across
  SU(2), SU(3), Spin(5), Sp(2) — **0 mismatches**, computed from the leading
  orbit with the bubbling deleted.                                          ∎

Both lemmas are statements about the **substrate** — measure, cocycle, Weyl
characters — and neither uses any property of the canonical basis beyond W2 and
`O(𝖖)`.  That is the analytic content the numeric check was standing in for.

**A corollary worth keeping, because it explains a known measurement.**  The
derivation shows `I[𝖖⁰]` *cannot see the bubbling at all* — every non-leading
cell is `O(𝖖)` by axiom and the weight cannot lower it.  So orthonormality at
`𝖖⁰` is **blind** to a candidate that is missing its bubbling, which is exactly
why the bar-blind `𝖖⁰` self-norm failed to catch the `pure_un_joint_fiber_defect`
offender while (★) rejected it 23/23.  The two facts are the same fact.

⚠ A label handed over as a **non-dominant representative** is not a distinct
label: at SU(3), `dominant_cochar_rep((1,0)) = (1,1)`, so `L_{(1,0),0}` and
`L_{(1,1),0}` are the same line, and a pairing between them is `1`, not `0`.
`fold` is what normalizes; a δ-test run on raw tuples will report spurious
off-diagonal ones (it did, here, before folding).

#### Closure — the one that is not derived

`span{L} ⊆ (★)-algebra` is immediate: (★) says each `L_{m,e}` is a `𝖖`-difference
operator preserving `Λ = R(G)`, and such operators are closed under composition.
So `L_a·L_b` always lies in the (★)-algebra, and the algebra axiom is **exactly**
the converse containment — in the user's words (2026-08-25, agreeing with the
characterization), *"the `L`'s span the (★)-algebra"*:

    (★)-algebra  ⊆  span_{Z[𝖖^{±1}]}{L_c}.

That is the same open question `residue_cancellation_recognition.md` §5 item 1
asks in its other guise ("a proof … that (★) + W1/W2 characterises the canonical
basis").  A natural route is a peel/induction: read the leading datum of a
(★)-element, subtract the matching canonical, and recurse — which terminates if
the leading data strictly descend in a well-founded order and the magnetic
support stays bounded.  Note that this is where the **support bound** would earn
its keep, so the two open questions of this session are connected rather than
independent.

Until it is proved, closure is **measured**, and the measurement must be on
products.  `decompose` peels top-down and would return a *partial* answer
silently if it ever stopped early, so the certificate is **exact
reconstruction** — `chart(a)·chart(b) == Σ_c C^c_{ab}·chart(c)` on the torus —
not merely "`multiply` returned something".  Battery:
`experiments/closure_products.py`.

### Memoizing the `L_{m,e}` — the chart memo and its persistence (2026-08-25)

**User direction.**  *"AbeKAlgebra has been simplified and improved.  We
temporarily deactivated various optimizations to build a solid general
constructor for `L_{m,e}` for every `(G,N)`.  I would like to use this session to
restore some of the optimizations (leaving the option to turn them off).  First
of all, obviously, we should memoize the `L_{m,e}` and make sure they can be
saved/loaded."*

The charts were already memoized **in memory, per instance**.  What was missing
was persistence and a public surface, and the tier now has both:
`AbeKAlgebra.save_cache(path)` / `load_cache(path)` — the same names and the same
shape as `RGKAlgebra.save_cache` / `load_cache`, so the chart tier's caches are
persisted the way the flow tier's already were.  A realisation opts in by
exposing `chart_cache()` (its live `{label: element}` memo) and `route_cache()`;
`nested_cache_algebras()` lets one file carry a delegated algebra's charts too,
which the `(G, N)` tier uses for its inner pure-`G` engine (ruling D3,
`Q[0⃗] = pure`, is read with it, so a matter cache without the pure charts would
still pay to rebuild them on the first `decompose`).

**The file is exact.**  A residual is an integer numerator over an explicit
denominator multiset, so a round trip is an identity, not a re-derivation
(`TorusLaurent.to_json` / `TorusRational.to_json`).  The `R[𝖖^±]` scalar path
honest-fails rather than flattening an `RElement` — writing one out needs the
ring's basis labelling in the header — and no shipped chart uses one.

**Two guards, because a cache file is untrusted input.**  It lets an `L_{m,e}`
enter the algebra without having been built by the guarded solve, and on this
tier a fast wrong answer is the dangerous failure mode.  So:

* **provenance.**  The header fingerprints the presentation: the torus shape, the
  root datum's structure, the line lattice's centre-class pairs, and the **phase
  convention probed rather than described** — `atom_phase_doubled` and
  `rho_sign_exp` evaluated on the simple coroots and their pairwise sums.  A
  datum's phase is a *callable* and cannot be compared any other way, and it is
  exactly what D31's odd-`⟨Σ⁺,m⟩` correction moves, so two data agreeing on
  structure and differing there produce different, both-well-formed charts.  A
  mismatch refuses the file.  This matters because a chart is a residual vector
  in **raw coordinates**: cross-loading SU(3)'s `m=(1,1)` into Sp(2) would not
  raise anywhere downstream, it would simply be wrong.
* **re-verification.**  Every admitted chart is re-run through the axioms —
  on the pure tier the full five-condition `star_bubbling.verify_axioms`
  battery, which includes the W2 seed *identity* and so catches a chart filed
  under the wrong label.  A failure raises rather than being skipped, because a
  file that disagrees with the axioms is evidence of a real problem and dropping
  the bad rows would hide it while the good ones loaded.

  This is affordable precisely because **verifying is not solving**: measured at
  ~5% of a rebuild (SU(3) `m=(2,2)` 0.28 s against 5.4 s; G₂ `m=(2,3)` 0.41 s
  against 6.3 s).  Trusting the file would buy 5% and give up the one guard that
  separates a canonical from a plausible impostor — and (★) is exactly the
  condition that rejected the `pure_un_joint_fiber_defect` failure mode 23/23
  where the `𝖖⁰` self-norm did not.  `verify="none"` exists and is not the
  default.

Measured end to end: SU(3) `m=(2,2)` **4.67 s to build, 0.27 s to load with full
verification**; G₂ `m=(2,3)` **5.69 s against 0.37 s**.  The matter tier's
`multiply` at SU(2)+2×2 goes 0.55 s → 0.011 s across a save/load.

Memoization is deliberately **not** a switch — the memo is structural (`_build`'s
cycle detection and `decompose`'s repeated `chart` calls read it), so an
off-switch would change behaviour and not merely cost.  What is offered instead
is `clear_cache()`, which reclaims it (nested sections included) and lets it
refill.  Battery: `tests/test_abe_chart_cache.py`.

### The declared optimizations — what came back, and what did not (2026-08-25)

**User direction.**  Stage 2 as D40 deferred it: *"restore some of the
optimizations (leaving the option to turn them off)"*.  The conclusion the
session reached, in the user's words: *"the θ-twist applied to general `(m,e)` is
a useful optimization, and other obsolete optimizations should be left maybe to
specific specializations of `AbeKAlgebra` which make assumptions on `G` and
`N`"* — *"and memoization of course"*.

So the **general** tier carries exactly the two optimizations that assume nothing
about `G` or `N`, and they compose: the chart memo (above) holds what has been
built, and the θ-twist turns each entry into its whole `m̄`-line.  Everything that
assumes a `G` or an `N` belongs to the class that already assumes it — the
2026-07-28 naming ruling applied to speed rather than to names.

The register is `PureGAbeKAlgebra.DEFAULT_OPTIMIZATIONS`, switched per name via
`optimizations=`; `optimizations=()` reproduces stage 1 exactly, so *"the axioms
alone build every label"* stays a **runnable** claim and not a historical one.
An unknown name raises rather than silently doing nothing.

| name | applies to | licensed by | measured |
|---|---|---|---|
| `theta_twist` | any label `T`-related to one already in the memo, plus the undressed anchor.  **Universal** — offered by both tiers | the `e ↦ e + k·m̄` symmetry `T^k : f_p ↦ v^{k·p̄}f_p`, MEASURED 30/30 and *not proved*, so the route stays (★)-guarded | identical element 14/14; 39× at SU(2) `m=(1,)`, 253× at SU(3) `m=(1,1)`, 1 773× at Sp(2) `m=(1,1)`, **18 370×** at SU(3) `m=(2,2)` (54.2 s → 2.9 ms), **29 540×** at G₂ `m=(2,3)` (63.8 s → 2.2 ms) |
| `monoid` | undressed `L_{m,0}` at a splittable dominant `m`.  **Pure gauge only** | a THEOREM: `L_{m,0} = (−1)^{⟨ρ,m⟩}𝖖^{⟨ρ,m⟩}Λ_m` with `Λ_m : P_μ ↦ 𝖖^{2⟨μ,m⟩}P_μ` on a common eigenbasis, so eigenvalues multiply and the normalizations cancel by linearity of `⟨ρ,·⟩` | identical 4/4; 277× at SU(2) `m=(4,)`, **345×** at SU(3) `m=(2,2)` (14.7 s → 43 ms) |

**The θ-twist is general in `e`, and that is the point.**  `T^k` carries
`L_{m,e}` to `L_{m, e+k·m̄}` at **any** `e`, not only at `e = 0` — the wiring that
only twisted the undressed anchor was a restriction of the law, corrected here
(user: *"every known `(m,e)` means we also know its θ-twists, not just `m,0`"*).
The electric labels at fixed `m` therefore fall into `m̄`-lines and one known
point gives the whole line, **in both directions**: measured reaching the
undressed `L_{m,0}` — the expensive charge — from a cached dressed neighbour at
`k = −1`.  This is what ties the optimization to the memo: it multiplies the
value of each *existing* entry rather than adding entries, and its coverage grows
as the memo fills.  `monoid` earns its place through the same composition — it
builds the undressed charge the twist twists *from*, so its win propagates along
the whole line above it.

`monoid` is **not** offered to the matter tier: a free monoid cannot carry
Littlewood–Richardson multiplicities, so `L^N_{m,0}·L^N_{m',0} = L^N_{m+m',0}` at
`(G, Adj)` would be *"very suspicious and suggests something badly wrong"* (user
ruling, 2026-07-30).  The name is still *accepted* there — turning it off must be
possible from the object the caller holds — and governs the inner pure-`G`
algebra, which genuinely is the `N = 0` theory.

**What did not come back, each on evidence.**

* `wilson` and `minuscule` are **subsumed**, as the user said before it was
  measured: at `m = 0` the tropical support is the single cell `{0}` so the
  system has no unknowns and the W2 seed `χ_e(v)·U_0` IS the answer; at minuscule
  `m` there is no bubbling, so the leading orbit is the canonical.  Measured
  identical 15/15 and 4/4, with the general route's overhead a **flat
  sub-millisecond constant** — its fixed setup, not a scaling cost, and paid once
  per label behind the memo.  Nothing to relocate.  (Until 2026-09-19 this
  sentence pointed at `PureUNKAlgebra` through `faster_equivalent()`; that
  class is retired and `faster_equivalent()` returns `None` everywhere — the
  measured picture is repo_audit A89.)
* `closed_form` is **excluded on a user ruling** — *"if closed_form is wrong
  occasionally, it is an indictment of closed_form"* — and the exclusion has a
  real measured cost, since it is 39–196× faster where it is right and reaches
  charges `monoid` cannot (those with no splitting).  At SU(3) `m=(2,2)` it
  returns a **wrong element without raising**: it fails W1 *and* (★), and the
  differing cells are exactly `W·(1,1)` together with the origin.  The mechanism,
  recorded so it can be fixed rather than only avoided: `one_step_cells` means
  one **root** below `m`, not one simple coroot, so `m − θ^∨ = (1,1)` is listed as
  a one-step cell although it lies two coroots down, the one-step residual is
  applied at an invalid depth, and the coverage check — which asks only whether
  each dominant cell is **present** — does not fire.  Its docstring's *"honest-
  fail, never a wrong element"* is withdrawn.  A route whose correctness lives in
  the guard rather than in itself is the failure mode the constructive-build rule
  exists to prevent.
* `cone` and `peel` stay retired: `cone` returns the bubbling-free cone monomial
  that (★) *correctly* rejects at non-minuscule `m` (the documented R7
  behaviour), and `peel` is the measured-expensive path (worst 95 s at
  `Sp(6)/Z₂`, with a form-aware variant measured **not** to help).

All of it stays reachable through `constructive_routes=True`, where output is
**compared** rather than trusted — the `recursive_spectrum` precedent, and what
makes agreement evidence rather than tautology.  Evidence, negative rows
included: `experiments/abe_optimization_benchmark.py`; correctness:
`tests/test_abe_chart_cache.py`.

### The spectral route — `L_{m,0}` in closed form (2026-08-25)

**User direction.**  *"If we can guess the general form of any `L_{m,0}` for any
`(G,N)` it would be great"*, and then — conditionally — *"if you can get it to
work and match the conventional constructor, 'promote the spectral law to the
production route for `L_{m,0}` at all `(G,N)`' would be great"*.

**The law.**  `L_{m,0}` is diagonal on a basis `{P̄_μ}` of `R(G)`:

    L_{m,0} = c_m·Λ_m,   Λ_m : P̄_μ ↦ 𝖖^{2⟨μ,m⟩}P̄_μ,   c_m = (−1)^{⟨ρ,m⟩}𝖖^{⟨ρ,m⟩}

with `{P̄_μ}` characterized **intrinsically** — triangular in the characters
(`P̄_μ = χ_μ + O(𝖖²)`) and orthogonal for `Δ = ∏_{α∈Φ}(v^α;𝖖²)_∞`, a Gram–Schmidt
entirely inside `R(G)`.  Nothing there references a chart or a solve, which is
what makes this a *construction* rather than an observation.  That much was
already in the repo (`spectral_eigenbasis` / `spectral_operator_matrix`).

**What landed: the transcription, and the route.**  The law delivers the operator
as a matrix on the **character basis**; the tier speaks the **f-presentation**.
`spectral_transcription.py` (spine, promoted from
`experiments/pure_g_transcribe_datum_only.py`, which now imports from it) bridges
them through

    Σ_a χ_μ(𝖖^{2a}v)·d_a(v) = Σ_ν C[μ][ν]χ_ν(v),   d_a = f_a(𝖖^a v)·ψ_a(v)

in two **decoupled** steps — pointwise in `{d_a(v)}` at exact rational points,
then per-cell numerator interpolation — because the joint system is 182 unknowns
at G₂ `m=(2,3)` and does not finish.  Cells are solved independently, so
cross-cell Weyl covariance is a *check*, not an input.  The ansatz is
**datum-only**: denominator `K_f(n)` ((★)'s own pole structure, the ansatz
`joint_solve` already carries), numerator support `box_points` (a cutoff,
certified by widening), `𝖖`-content palindromic (= W1).

Exposed as `PureGAbeKAlgebra(..., optimizations=("spectral",))`, guarded by
exactly what `solve_canonical` applies (W1, (★), the leading orbit) and refusing
rather than returning a candidate.

**It matches the conventional constructor** — the condition the promotion was
made on.  Every cell `== solved chart` on SU(2) `m=1,2`, SU(3) `(1,1),(1,2)`,
SU(4) `(1,1,1)`, Spin(5) `(1,1),(2,1)`, Sp(2) `(1,0),(1,1)`, G₂ `(2,3)`; W1, (★)
and the leading orbit `ok` on all of them.  And the agreement is **evidence**, not
a restatement: `spectral_canonical` calls no `solve_canonical` anywhere in its
path.

**Cost — why it is OPT-IN and not a default.**  It is not uniformly faster, and
at shallow charges it is much slower:

| | spectral | (★) solve |
|---|---|---|
| SU(2) `m=(2,)` | 0.96 s | 0.03 s |
| SU(3) `m=(1,1)` | 1.57 s | 0.09 s |
| Sp(2) `m=(1,1)` | 1.92 s | 0.77 s |
| SU(3) `m=(2,2)` | 30.2 s | 11.5 s |
| G₂ `m=(2,3)` | 122 s (69 s of it the Gram–Schmidt) | 13.1 s |
| **Sp(2) `m=(2,1)`** | **>25 min (unfinished)** | **132.9 s** |

⚠ **No crossover has been found, and an earlier draft of this section overstated
the trend.**  It read "the gap narrows with depth", extrapolating from the first
two rows (32× → 2.6×).  The last row shows that pattern does **not** continue.
The honest statement is that the spectral route has **never been measured faster
than the (★) solve at any charge**, and at the deepest charge tried it is
dramatically slower.

The *structural* argument for an eventual crossover is kept, because it is what
would make one plausible: the solve grows superlinearly in bubbled cells carrying
free parameters, while the spectral cost is dominated by a Gram–Schmidt depending
on `(datum, height, order)` and **not** on `m` — memoized, so it amortizes across
every charge at one datum.  But an argument is not a measurement, and the
measurement has gone the other way at every charge tried.

**So choose this route for what it IS, not for speed**: a closed form, and a
construction independent of the (★) solve.  Turn it on when you want the
independent cross-check or the spectral description; it is not currently a
performance option for anything.

**The `(G,N)` half — LANDED for the theories below (2026-08-25).**  `matter_spectral_transcription.py` carries it, and the law needed
two changes and no others: `{P̄^N_μ}` is Gram–Schmidt against
`Δ(G,N) = Δ(G)/∏_{i,w}(−𝖖μ_i v^w;𝖖²)_∞` instead of `Δ(G)`, and **`c_m` is left
exactly as in pure gauge** — the eigenvalues do not move, matter enters only
through strictly-lower μ-carrying corrections.  So the right-hand side needs only
`Φ`, `wt(N)`, and the pure `c_m` (taken from `spectral_diagonal` at `μ = 0`, not
read off a measurement — that is what keeps it a construction).

*The transcription is the pure one, one μ-level at a time*, which is the measured
fact that makes it tractable rather than a new problem: 96/96 over (point,
μ-level, weight) triples.  The per-cell ansatz transports with exactly one new
ingredient, all three parts measured **before** being coded:

| part | rule | measured |
|---|---|---|
| denominator | the **pure** `K_f[a]`, unchanged | 39/39 |
| numerator box | the pure box **anchored at `e_k = Σ_i k_i·λ_i`** | 72/72 |
| `𝖖`-content | palindromic **per μ-level** | 39/39 |

The anchor has a reason rather than being a fit: `box_points` already anchors at
the Wilson weights of its `e` argument, and a μ-level `k` has consumed `k_i`
quanta of matter irrep `i`, moving the numerator onto the root-lattice coset
those weights determine.  Any weight of the irrep gives the same coset, so the
highest weight is canonical — and `pad` does **not** reach it, since `pad`
dilates along root directions only (D35), which is why this needed a rule and not
a wider box.  The test asserts the anchor is load-bearing (6 monomials fall
outside the unanchored box).

`F` is what the transcription recovers; the chart is `Q = F/Z`
(`divide_by_Z_vec` — arithmetic, not a solve), because the substrate's own
cocycle already carries the dressing.

**Reproduces the conventional constructor** at SU(2)+1×fund `m=(1,),(2,)`,
SU(2)+2×fund, SU(2)+3×fund, Spin(5)+1×spinor, Sp(2)+1×fund, **SU(3)+1×fund
(4.2 s)** and **SU(3)+2×fund (24.9 s)** — the last two being the non-self-dual
cases that the fixes above opened.

**A truncation certificate is mandatory here, and it is where the cost sits.**
`Δ(G,N)`'s matter factor is a *geometric* expansion, so the `𝖖`-truncation bounds
it, and `λ̄_μ` then shifts the series **down** by `2⟨μ,m⟩`.  Too small an order
does not fail loudly — it adds terms at the **bottom** of each series, on top of
the correct ones.  Measured at SU(3)+1×fund `m=(1,1)`: at `𝖖`-order 14 the
legitimate μ-levels agreed on only 18/29 entries, every discrepancy an added
`𝖖^{−9}`/`𝖖^{−11}`/`𝖖^{−13}`, while SU(2) was exact at the same setting.  The
transcription cannot tell that from signal (it evaluates whole rows numerically),
so it reports an inconsistent pointwise system — safe but useless.  The
certificate recomputes one step wider and demands the identical matrix, exactly
as the pure route's does.

**Two defects in the measure, found by asking whether the orthogonality actually
holds (user, 2026-08-25) — and both invisible on every theory the machinery had
been exercised on.**  An earlier draft of this section reported `SU(3)` +
fundamental as a *cost boundary* ("did not certify within ~20 minutes").  That
attribution was **wrong**: it was these defects.  With them fixed, `SU(3)`+1×fund
builds in **4.2 s** and `SU(3)`+2×fund in **24.9 s**, both equal to the
conventional constructor.

Both hid for the same structural reason: every matter rep previously used — the
fundamental of `SU(2)`/`Sp(2)`, the spinor and vector of `Spin(5)` — is *both*
multiplicity-free *and* self-dual.

1. **`Δ(G,N)` dropped weight multiplicity.**  Its hypermultiplet factor is one
   `(−𝖖μ_i v^w;𝖖²)_∞` per weight **occurrence**, but the callers passed the
   *distinct* weights.  The **adjoint at rank ≥ 2** is the first rep with a
   repeat (`SU(3)`'s adjoint has the zero weight with multiplicity 2 = rank), so
   `Δ(G,Adj)` was built with one Cartan factor instead of two — and the adjoint
   is exactly the physically important case (`N = 2*`, the Langlands family
   `(G,Adj) ↔ (G^∨,Adj)`).  `matter_weight_multiset` now returns one multiset
   **per slot**, since slot `i` carries `wt(N_i)`.

2. **The Gram matrix assumed the pairing is symmetric.**  `⟨f,g⟩ =
   CT_v[f(v⁻¹)g(v)Δ(G,N)]` is symmetric **iff `Δ(G,N)` is `v→1/v` invariant, iff
   `wt(N)` is self-dual** — the numerator `Δ(G)` always is (it runs over all
   roots), but the matter factor runs over `wt(N)`.  The **fundamental of
   `SU(N ≥ 3)` is not self-dual**: measured, the pairing differs there on 10/12
   ordered pairs, so filling `G[(a,b)]` from `G[(b,a)]` put the transpose in half
   the matrix and the resulting basis satisfied **no** orthogonality (0/3, 0/15;
   computing both orders gives 3/3, 15/15).  `pairing_is_symmetric` is the cheap
   datum-only predicate for which regime one is in.

**And the asymmetric case is genuinely two-sided.**  With an asymmetric pairing
the two one-sided Gram–Schmidts give *different* bases, and only one is the
eigenbasis of `L^N_{m,0}` — measured, not chosen: putting `P_μ` in the **second**
slot (`⟨χ_ν, P_μ⟩ = 0` for `ν < μ`) reproduces the built operator **23/23** at
`SU(3)`+fund, while the first slot manages 15/30.  For self-dual matter the two
coincide, which is why the choice was invisible.  This is the `P`/`Q`
bi-orthogonal situation of Macdonald theory showing up where the matter is not
self-dual.

**Where the measure sits in the standard vocabulary.**  At **adjoint** matter
`wt(N) = Φ ∪ {0}`, so the hypermultiplet factor runs over the *roots* and

    Δ(G, Adj)  =  Macdonald density(q = 𝖖², t = −𝖖μ)  ×  (−𝖖μ; 𝖖²)_∞^{rank}

— measured exactly, **175/175** `v`-monomials at `SU(3)`, 15/15 at `SU(2)`.  The
extra factor is the Cartan (zero-weight) part and is **`v`-independent**, and
Gram–Schmidt is blind to an overall constant, so `{P̄^{Adj}_μ}` *are* Macdonald
polynomials with the flavour fugacity as the Macdonald parameter `t`.  (The pure
case is `t = 0`, q-Whittaker — §7l.)  For a general `N` the denominator is
indexed by `wt(N)` rather than by the roots, which is outside the standard
families; that is the honestly-generalized case.

**Still open (not attempted):**  rank-2 gauge groups with matter (the cost
boundary above); the deep charges where the ~400× is expected to live (Sp(4)
`m=(2,1)`: the conventional route takes 934 s there); and the **dressed**
direction `L_{m,e}` at `e ≠ 0`, for which §7n of `pure_g_final_form.md` derives
the shape — on the `𝖖`-commuting class `L_{m,e} : P̄_μ ↦ s(μ)·P̄_{μ+e}`, a
monomial matrix, the same order of data as an eigenvalue — but leaves both the
criterion for which `(m,e)` qualify (three candidates refuted) and the scalar
`s(μ)` open.

### Instances (all certified)

| instance | labels | decompose engine | certification anchors |
|---|---|---|---|
| **(2026-09-19)** the four type-A rows below — `PureUNKAlgebra`, `UNNfKAlgebra`, `UNQuiverKAlgebra`, `PureSUNKAlgebra` — are RETIRED to `legacy/` (user ruling; measured against the general classes and adversarially reviewed, repo_audit A89; `PureSUNKAlgebra`'s oracle transport was wrong at dressed 2·adjoint labels, A75).  Their roles are `PureGAbeKAlgebra(u_n(N) / su_n(N))` and `GNAbeKAlgebra` (U(N)+N_f through the D8b seam; linear chains on `root_datum.product_datum`, `tests/test_gn_abe_quiver_chains.py`).  The rows are kept as the record of what they were. | | | |
| `PureUNKAlgebra` (keystone) | engine `(m, λ)` | the registry-image leading-orbit peel — recognize on residuals, **normalize by the image's own leading monomial in the same read frame** (dressed labels carry a Witten-frame q-power; the loop diverges without this), subtract `urqt(label)`, recurse | derived ≡ engine on multiply/ρ/ρ⁻¹/trace/inner product (`tests/test_pure_un_abe_contract.py`); the full keystone battery |
| `UNNfKAlgebra` (all N_f) | `(m, λ)` at N_f=1; `((m, λ), w)` with `w` a dominant SU(N_f) weight over `R(SU(N_f))` (`zplus_ring.SUNZPlusRing`) | per-level pure registry peel → the certified level-ascending `matter_decompose` attribution → the **SU peel** (central flavour fugacity specialized to 1 — a central specialization is an algebra homomorphism, so associativity/bar/orthonormality survive; ruling D8b) | vs the flavour-collapsed / D8b-projected `UNNfOverPure` (meson seam, SU-character traces = the T5 `U(2)=SU(2)×U(1)` finding at ring level, orthonormality, BPS-certified vacuum) |
| `UNQuiverKAlgebra` (linear U(N_i) chains) | tuples of per-node `(m, λ)` (link U(1)'s specialized per link — same homomorphism argument) | the **joint registry peel**: `_joint_recognize_leading` + tensor registry images + the per-node lead normalization, then lowest-level-first attribution | vs `QuiverOverPure` link-collapsed at U(2)×U(1) / U(2)×U(2) / U(1)³ (the bifund meson `E·F = L_{(1,−1)} + q·L_{(det₁,det₂⁻¹)}`, cross-node screening, trace equality) |
| `PureSU2KAlgebra` (pure SU(2)) | `(m, e)` in the `su2_fold` frame | the generic `wrq_torus` peel; **fully native** — the adjoint-monopole fiber is the single closed-form input, then product-and-peel + `kl_bar_correct` (Plan 24, #765) | `==` the BPS Kronecker chart (`pure_su2_bps_iso`) on multiply/ρ/trace/orthonormality; `well_formed` on every chart |
| `PureSUNKAlgebra` (pure SU(N)) | ambient `(m, e)` on the trace-zero lattice | registry peel on `SUNRQTorus`, charts from the `PureUNKAlgebra(N)` oracle restricted to trace-zero + det-collapsed | SU(3) certified incl. orthonormality (#766); SU(3) adjoint bubbling closed-form (#785) |
| `PureGAbeKAlgebra` (pure gauge at **any** `RootDatum`) | `(m, e)`, `m` cochar-dominant / `e` Levi-dominant | the **licensed (★) solve** (`star_bubbling`) at every label, `m = 0` included (D40) — beside it the two **declared optimizations** `theta_twist` and `monoid`, each guarded and each asserted to return the *same* element (D43); `optimizations=()` is the bare axiom route.  `route(label)` reports which fired.  Charts are memoized and the memo is persistable | at `su_2()` `==` `PureSU2KAlgebra`'s native build and at `su_n(3)` `==` `PureSUNKAlgebra` (the independent-route rows that certify the solve); beyond type A self-certified by (★) + W1/W2 + bar + ρ + orthonormality (`tests/test_pure_g_abe_kalgebra.py`) |
| **`GNAbeKAlgebra`** (`(G, N)` — gauge theory with `T^*N` matter at **any** 4d gauge group and **any** matter rep; Plan 24 TM6.  Named by user ruling 2026-07-28; `g_matter_abe_kalgebra.py` is a shim.  The **4d gauge group data** is the `lines=` argument — a `global_form.LineLattice`, i.e. a *maximal set of mutually compatible Kapustin `(m, e)` labels*, the same convention `PureGAbeKAlgebra` takes and NOT "which cocharacter lattice" (that is the 3d reading, `threed.root_data.GaugeDatum`); handed through to the inner pure algebra so there is one admission gate.  **Non-simply-connected forms BUILD** — rulings D10/D19/D31; the earlier "honest-fail, buildable only in the standard-`Z²` BPS chart" reading is RETRACTED: PSU(3)/PSU(4)/PSU(5)/Sp(6)/Z₂ carry their defining fractional coweights, intermediate forms like SU(4)/Z₂ build, and odd `⟨Σ⁺,m⟩` is not an obstruction either) | `((m, e), w)` — the pure-`G` Kapustin charge plus a flavour irrep `w` of `∏_i U(n_i)` (ruling TM7: `n_i` copies of the same irrep are interchangeable, so they carry `U(n_i)`, not `n_i` separate `U(1)`s) | the **quotient `Q`** of the guarded constructor's `F = Z·Q` (`matter_star_bubbling` / `matter_multislot`) — the substrate's own cocycle `W = T(Z)T(Z)/Z` already carries the dressing, so presenting `F` would count it twice.  `decompose` is level-ascending: `Q[0⃗] = pure` (ruling D3) makes the lowest μ-level slice a **read** by the pure-`G` engine, then subtract and recurse | two disjoint legs — `Z·chart == GMatterOverPure.rg_chart` (SU(2)/U(2)/Sp(4)/Spin(5)-spinor) and `chart == UNNfKAlgebra.chart` in the type-A corner (an independent oracle with no flow and no `matter_star_bubbling` in its path); plus reconstruction, positivity/integrality of 300+ structure constants (`tests/test_g_matter_abe_kalgebra.py`) |
| named `(G, N)` **presets** (`implementations/g_matter_roster.py::ROSTER` + `roster(name)` / `roster_spec(name)`) | — | **parameter tuples, not classes** (user ruling 2026-07-28: *"use special names only if you have algorithms specifically optimized for a G and/or N"*).  A special name is earned by an algorithm, and the classes that earned it already exist — `PureUNKAlgebra`, `UNNfKAlgebra`, `UNQuiverKAlgebra`, `PureSU2KAlgebra`, `PureSUNKAlgebra` — reachable from an instance via `faster_equivalent()`.  An earlier per-theory subclass roster was pure delegation and was retired by the ruling | all 13 presets certified at identity + Wilson + minimal monopole (W1, `certify_canonical`, ρ round-trip), 12 of them sub-second; `g2-nf1`'s monopole builds in ~145 s — a **cost** observation, not a wall.  Also pinned: the 4d gauge group data is *maximal* + *mutually compatible* (`verify_maximal` / `verify_mutually_local`), and a non-simply-connected form honest-fails (`tests/test_g_matter_roster.py`) |

**The (★)-guarded solve (user ruling 2026-07-27).**  The tier's
constructive-build rule turns on there being *no post-hoc guard*; **(★) — the
affine-Weyl residue cancellation — is that guard**, so a solve carrying it is
licensed.  Its content is an axiom: Neumann lines are generated by
`L_{0,e'}|N]`, and since `L_{0,e}|N] = χ_e` exactly, `span{L_{0,e'}|N]} = R(G)`,
so **(★) ⟺ the element's `𝖖`-difference operator preserves `R(G)`** — a
membership condition on the presentation, not a pairing inside the span (which is
why it rejects the `pure_un_joint_fiber_defect.md` failure mode 23/23 where the
`𝖖⁰` self-norm did not).  **The tier still exposes no solve entry point**: the
license lives in `star_bubbling.py`, which applies unique-and-verified-over-`Z`
+ W1 + (★) + a pad+1 box certificate before returning.  This is what opens the
groups with **no minuscule cocharacter** (`Spin(5)`, `G₂`, `C₃`), where
product-and-peel has nothing to start from.  Rulings:
`restructuring_plans/24_nonminuscule_groups/decisions.md` D5/D6.

A minimal pure-U(1) harness instance (`tests/test_abe_kalgebra_contract.py`)
pins the tier itself: the triple alone yields the quantum-torus constants,
associativity, ρ roundtrips, the engine-ground-truth trace, and the
orthonormality sweep.

### Flavour rings (rulings D8 / D8b)

The faithful flavour symmetry of a U(N) node with `M` fundamentals is
**SU(M)** — the matter is organized as (fundamental of U(N)) ⊗
(anti-fundamental of SU(N_f)); the diagonal U(1) and the link U(1)'s sit
inside gauge centres.  The flavour Wilson μ and the gauge det Wilson are
**not** interchangeable one-for-one (label-dependent q-powers,
bar-inconsistent — measured), so the central direction is removed by
**specializing the central fugacity**, never by identifying labels.
`R(SU(N))` is `zplus_ring.SUNZPlusRing` (trimmed-partition basis,
Kostka-DP Littlewood–Richardson; the `sun_characters` Weyl-denominator
machinery is the cross-certifying oracle and the flow-level
recognize-after layer — `sun_flavour_enhancement`).

### Open items

The fully native constructive **build on the product lattice** (today's
quiver shells take charts from the certified flows — Route-A provenance,
exactly as the retired `PureUNKAlgebra` predated its closed-form engine); per-node
fundamentals on quivers (compose the SU peel per node); BPS chart-graph
(multi-chart) refinement; the BPS quiver-dictionary cross-check and the
U(3) iso leg.  History and rulings: `restructuring_plans/30_abe_kalgebra_contract/`.

**Note (2026-07-28): the general-*matter* case is no longer open.**  The
`(G, N)` ladder closed through the native tier (Plan 24 TM1–TM6 +
`GNAbeKAlgebra`, TM7 flavour, TM8 the roster / removal flows / type-A seam),
so "per-node fundamentals" above is now specifically about **quiver** nodes; a
*single* node at any `RootDatum` with any matter representation is done.

### The two flavour seams of the `(G, N)` tier

Three conventions for the same flavour data are live, and they are related by
`RingHom`s rather than by identification (which is what ruling D8b insists on):

| convention | who uses it | ring |
|---|---|---|
| one `U(1)` per hypermultiplet slot | the **flows** (`GMatterOverPure` / `GMatterOverMatter`) — the μ-levels are what the RG cone is graded by | `AbelianZPlusRing(M)` |
| `R(U(n_i))` per group of identical irreps | the **native** class `GNAbeKAlgebra` + the roster (ruling TM7) | `⊗_i UNZPlusRing(n_i)` |
| `R(SU(N_f))` | `base_change(un_to_sun_hom(N_f))` of the native class (the purpose-built type-A `UNNfKAlgebra` that carried it natively — rulings D5/D8b — was retired 2026-09-19) | `SUNZPlusRing(N_f)` |

* `zplus_ring.un_to_cartan_hom(n)` : `R(U(n)) → R(U(1)^n)` — the Cartan
  restriction (the weight diagram; the `U(n)` analogue of `su2_to_u1_hom`).  This
  is the native↔flow dictionary, and comparing through it is what makes the
  Goal-1.3 statement "the flow's UV and the native class are the same algebra"
  *exact* rather than augmentation-blurred.
* `zplus_ring.un_to_sun_hom(M)` : `R(U(M)) → R(SU(M))` — the **central
  specialization** (`det ↦ 1`; `TrivialZPlusRing` at `M = 1`).  This was the
  native↔`UNNfKAlgebra` seam (`g_matter_un_nf_seam.UNNfSpecialization`, in
  `legacy/` since 2026-09-19 with that class; the agreement it measured —
  406/406 at (2,1), (1,3), (2,2) — is recorded in repo_audit A89).

**That last seam is a surjection, not an isomorphism**, and the three tempting
stronger readings are all measurably false — see the module docstring; each is
pinned by a test.  The sharpest of the three is worth stating here because it is
easy to get wrong: even at `N_f = 1`, where structure constants and ρ *do* agree,
the **trace** distinguishes them.  The matter `μ` grades matter zero modes, so it
appears in the trace of a flavour-*neutral* canonical (the `U(2)+1` Wilson line
has `Tr = −μ⁻¹𝖖 + 2μ⁻¹𝖖³` — the Wilson line screens against the matter and picks
up flavour charge), whereas `UNNfKAlgebra` specialized that fugacity to 1 and a
spectator `U(1)` adjoined by `add_flavour` is inert.  So the difference between
the two type-A classes is a genuine **μ-refinement of the index**, not a
spectator flavour that could be tensored back on.

## BPSKAlgebra public surface

```python
A = BPSKAlgebra(pairing=B, node_charges=[...], spec=...)   # spec optional

# KAlgebra contract
A.identity(); A.multiply(a, b); A.rho(a); A.rho_inverse(a)
A.trace(a, K); A.inner_product(a, b, K)

# RG-flow output
A.F(a)                      # F(L_a) = dict{ γ (QT index): LaurentPoly } — a is a canonical label, keys are charges γ
A.spectrum_generator(K)     # diagnostic: S truncated to q^K  (LaurentPoly, 𝖖-truncated)
A.spectrum_generator_from_factors(cutoff, order=..., phases=..., leading_data=...,
                               with_multiplicities=False)
                            # S from its LEADING DATA as a product of BPS factors —
                            # spec-free, F-free (bps_factor_spectrum.py).  Exact
                            # HabiroElement coefficients, truncated only along the
                            # CONE.  A different truncation from spectrum_generator(K),
                            # not a drop-in for it.  with_multiplicities also returns Ω.
A.verify_spectrum_generator_from_factors(cutoff, order=...)
                            # that S == this chart's own [S|0⟩]_γ, in-cone

# choosing the order (module-level, bps_factor_spectrum.py) — OPT-IN, no default changed
# WHAT THE ENGINE REQUIRES IS A TOTAL ORDER ON THE POSITIVE CONE, nothing more:
# each BPS factor may sit at an arbitrary position among the pre-existing ones.
# A central charge is one PARAMETRIZATION of that, and a narrow one — at rank 2 a
# linear Z reaches 2 orders against the 43 free insertion finds at Kronecker-2.
BPSFactorSpectrum(..., order_key=lambda k: ...)
                            # ANY total order on the cone (node coords -> sortable);
                            # the general path.  Mutually exclusive with phases=.
                            # ⚠ keys on γ, so all of γ's spin pieces go to one
                            # position; per-(s,γ) placement is open, §27
BPSFactorSpectrum(..., phases=[...])
                            # the LINEAR parametrization: arg Z_γ per node.
                            # SUPPLYING IT is what asks for a central-charge
                            # order — the only way one is ever used
BPSFactorSpectrum(...)            # DEFAULT: no central charge at all.  order='strip'
                            # (strip node order + mean_rank_key) on an acyclic
                            # quiver — `rank` rays, spin-0 only, same chamber a
                            # good central charge reaches; order='random' (each
                            # (s,γ) piece at its own position) where the strip
                            # leaves a core
bps_factor_spectrum.mean_rank_key(node_order)
                            # phase-free cone extension of a node order:
                            # (Σ k_i r_i)/(Σ k_i), exact Fractions.  A mean has
                            # the betweenness property that is the ONLY thing a
                            # linear Z contributed to the order
bps_factor_spectrum.acyclic_node_order(pairing, node_charges)
                            # the nodes' placement ORDER from the source/sink
                            # STRIP — the minimum-ray chamber (`rank` rays) on an
                            # acyclic quiver.  Returns None when the strip leaves
                            # a CORE: there it degenerates to the identity and
                            # measures WORSE than a random permutation, so None is
                            # the honest answer.
bps_factor_spectrum.central_charge_for_node_order(order)
                            # extend a node order to the whole cone LINEARLY ->
                            # a phases= vector.  Only for callers who want the
                            # classical phase order; mean_rank_key does the same
                            # job without any complex numbers.
bps_factor_spectrum.source_sink_strip(bracket)
                            # (placement order, core size); the strip itself.
                            # Deliberate copy of bps_quiver_tools's mantle strip,
                            # pinned equal to it by tests/test_ray_spectrum.py

# charge ↔ label (free-R-module API + tropical labels)
A.embed_R(r)                # central embedding R ↪ A: μ^f ↦ L_{γ_f} (γ_f ∈ ker B); 1_R ↦ identity
A.section_decompose(a)      # (section_rep, r_coeff): the R-module coordinate of L_a
A.gamma_lower(a)            # γ₋(a) = a (lower tropical charge)
A.gamma_upper(a)            # γ⁺(a) = −σ⁻¹(a) (upper tropical charge)
A.tropical_interval(a)      # (γ₋(a), γ⁺(a)) — F(L_a)'s support lies in [γ₋, γ⁺]
A.verify_F_S_leading(a)     # discovery relation F_γ S|0⟩ = X_γ + O(q)
A.verify_embed_section_roundtrip(a)   # embed_R(r_coeff(a))·L_{section(a)} == L_a

# spec management
A.shorten_spec()            # → new BPSKAlgebra with shortened spec via local moves

# inherited verifiers
A.verify_*(...)
```

The chart graph is **private state**: methods like `_solve_F_via_chain`,
`_multiply_via_chart_search`, `_solve_F_at_chart` are internal
optimizations that do not appear in the public surface.

### Construction contract

The constructor enforces:

1. **Non-degenerate pairing** (always-on, regardless of `verify` mode).
   Flavoured theories should be gauge-projected upstream before
   construction.
2. **Pointed positive cone**.  `verify="lazy"` (default) defers the
   expensive search; `cone_witness=...` short-circuits the search but
   not the assertion; `verify="off"` skips the search entirely (caller
   takes responsibility).

When `spec` and `negating_sequence` are both `None`, the constructor
auto-finds a spec via `BPSQuiver.find_negating_sequence(strategy="auto")`
(`find_spec_auto`, `s_spec_finding.md` §6/§9).  Since 2026-09-23 (Plan 41 D37)
the quiver is first split into its **strongly connected components** — maximal
node sets each of whose nodes reaches every other along arrows — taken in a
source-first order, every arrow between two components pointing forward; a
single node is mutated once, each larger component is searched on its own, and
the pieces concatenate into a negating sequence (a theorem:
`restructuring_plans/41_enumerated_quiver_dictionary/strongly_connected_redesign.md`
§1.1).  So an acyclic quiver needs no search, and the cost follows the largest
component rather than the rank; a large strongly connected component can still
take minutes — supply `spec` directly when you have one.  The auto-found spec
is not claimed shortest (user: composed specs are acceptable at any length,
D39); `shorten_spec=True` post-processes it by local moves.

## Algorithm survey

### S-finding — the crystalline factor engine, and the retired peel engine

`S` need not come from a spec.  Two constructions exist on the canonical surface
and they share no mechanism, which is what makes their agreement evidence rather
than a self-check — **but only one of them is active.**

**The factor engine is the default** since 2026-08-12 (user: *"replace the peel
engine (but leave it accessible)"*) and since 2026-08-13 the **only active
engine**: the peel route is **RETIRED** (user: *"The peel algorithm to build `S`
seems a bit antiquated now.  Retire it temporarily (but do not erase it)"*).
Retired means the ways *in* are closed — `recursive_spectrum.
build_spectrum_generator`, `engine="peel"`, `build_S_engine="peel"` all raise
`RetiredEngineError` — while every line of the recursion stays, reachable by
`allow_retired=True`, the `enable_retired_peel_engine()` context manager, or
`PEEL_RETIRED = False`.  Gating rather than deleting is deliberate: the peel
recursion is the repo's only *independent* construction of `S`, so keeping it
switchable keeps the cross-check runnable.

| engine | status | mechanism | truncation | fails when |
|---|---|---|---|---|
| **BPS factors** — `bps_factor_spectrum.build_spectrum_generator_from_factors`, `BPSKAlgebra.spectrum_generator_from_factors` (`pronilpotent_group_conjecture.md`) | **active, default** | prescribe the **leading data** (`𝖖¹`-coefficient `−1` on the nodes, `0` elsewhere) and read the palindromic factor multiplicities `Ω_γ` off degree by degree | cone degree | never structurally — there is no gate.  What is conjectural is *surjectivity* of the factorisation, so the output is cross-checked against an independently computed `S`, not assumed |
| **peel** — `recursive_spectrum.build_spectrum_generator[_auto]` (`recursive_spectrum_generator.md`) | **retired 2026-08-13**, intact behind one flag | remove a node, solve `F_γ · S_sub = X_γ + O(𝖖)`, reattach `E_𝖖(F_γ)` | cone degree | the peel is a **character ray** (matter): the monomial-charge gate trips and it honest-fails.  N=2\*/Markov and wild quivers are out of reach — the reason for both rulings |

**Not retired, despite living in the peel engine's module**:
`extract_spec_from_quiver`, `build_spectrum_generator_auto` and
`principled_sigma_maps` are engine-agnostic scaffolding — they *consume* an `S`.
And `BPSQuiver.build_spectrum_generator` in `bps_quiver_tools` is a **different
object** despite the identical name: the cluster-side spec builder (negating
sequence → spec), fed by the bidirectional-BFS `find_negating_sequence`, which is
the **cluster definition of the DT invariant** and is explicitly retained (user,
2026-08-13) — and which, since 2026-09-23 (D37), searches each strongly connected
component on its own and concatenates, under either strategy.

Measured on 18 quivers (`experiments/s_ray_spectrum_benchmark.py`): the peel
engine gates out on 4, the factor engine builds all 18, and on the 14 both build
`S` agrees **exactly**.

**Searching the order — `factor_order_search.py`.**  `S` is order-independent but
its *factorisation* is not, and some orders factor it far more simply than
others.  `find_simple_factorisation(pairing, nodes, cutoff)` runs a BFS over ray
factor **insertion points**, layered by cone degree and ranked by an **overall
cost** — the number of terms, with a dislike for `s > 0` — looking for a factorisation with **spin 0 only** that **stops populating**
inside the cone, which is what consumers need (user, 2026-08-13: a *true* spec's
green/red condition is not required; and *"a 'spec' `S` will just stop populating
at some point"*).  Suppressing spin is the heuristic because an `s > 0` factor is
believed to imply infinitely many more.  Two exact prunes (a forced `Ω` with
nonzero spin, or a negative multiplicity, kills the branch permanently) and one
heuristic (`keep_per_degree`, reported through `Result.exhaustive`).  On an acyclic quiver it short-circuits to the
strip order, which already attains the provable floor of `rank` factors.  Where
the strip leaves a core it is the whole point: 13–29 rays with spin up to
`2s = 14` collapse to 4–8 spin-0 factors, agreeing in length with both
`extract_spec_from_quiver` and the cluster-side `find_negating_sequence`.  Every
reported factorisation is rebuilt by the independent sum route before it is
reported — `Theory.S_from_spec` for a spin-0 spec, and since 2026-08-14
`nahm_local.general_nahm_habiro` for a content carrying spin, so the check now
covers **every** route the search can return rather than the spec ones only.

**Accepting a factorisation takes a WIDER cone, not a wider margin.**  An in-cone
rebuild plus "the factors stopped early" is measurably *not* a certificate: it
admits **impostors** — contents exact at every degree checked and wrong at the
next.  Two are pinned (`repo_audit.md` A53), and the sharp one is the 5-cycle at
`D = 6`, where a 9-factor impostor and the valid 8-factor answer top out at the
**same degree**, so no stopping margin can separate them.  So `is_spec` requires
the product to reproduce `S` on a cone strictly wider than the one that produced
it (`CONFIRM_EXTRA`), and `STOP_MARGIN` is demoted to the cheap screen deciding
whether that confirmation is worth paying for.  The rule this replaces —
"calibrate the margin" — failed twice; the corrected one is **when a check can
only see inside a truncation, check outside it**.

⚠ **And a wider cone at a FIXED depth is not a certificate either** (`repo_audit.md`
A56, 2026-08-14, the day after the above): `CONFIRM_EXTRA = 1` was refuted the same
way the two margins were — at cycle6, `D = 7`, two draws both granted `is_spec`,
one exact at every degree through 10 and the other wrong at 9.  Three fixed depths
have now fallen, so no constant settles it and none is claimed: `is_spec` is a claim
**relative to `Result.confirmed_to`**, the degree the run actually paid to check,
and `confirm_extra=` buys more.  Read the depth, not the boolean.
Where **no** spin-0 factorisation exists it returns the sparsest order instead of
nothing (Markov 16 rays → 7), because a sparse `Ω` makes downstream calculation
cheaper and is an optimization goal in its own right (user).

**Factoring an `S` you already have.**  `simplify_factorisation(S, pairing,
nodes, cutoff)` is the same search entered from the other end (user, 2026-08-13:
*"an algorithm to find the simplest factorization of an already-computed `S`.
Simplifying could be useful even at intermediate stages, before pushing a cutoff
higher"*).  The engine asks an order for one number per charge — the prescribed
`𝖖¹`-coefficient — so an existing `S` **supplies its own leading data**
(`leading_data_from_spectrum`), and nothing about its provenance is needed.
`verify=True` by default is not a formality: handed one quiver's `S` against
another's pairing, `verify=False` returns a valid spec **of the wrong quiver**,
silently.  For the intermediate-stage half, `seed_order=` takes a previous
`Result.piece_key()` and replays it as far as it stays clean before handing over
to the BFS — donated, not trusted, since a good low-cutoff order replayed
verbatim one degree up was measured to fail 4 times in 5.

**Which route to reach for.**  `extract_spec_from_quiver` is the cheaper tool
whenever *any* finite chamber will do — it consumes an `S` and recovers factors
by insertion.  The order search is for asking *which* orders give a simple
factorisation, and for the quivers where the default order gives a bad one.  It
is also, on two measured quivers, the crystalline route that finishes at all: the
extractor exceeds a 60 s box on **A6 linear** and **5-cycle(1)** (2.7 GB RSS
before being killed, unbounded), where the search answers in 0.10 s and 4.84 s.
None of the three routes dominates.

**On cost, the honest statement is that neither engine dominates, and the
variable is the CONE SIZE rather than the rank.**  The factor engine places one
factor per ray and the ray count grows with the cone `C(D+r, r)`; the peel engine
does ~`rank` `F`-solves.  Measured over a (rank, degree) grid
(`s_leading_term_existence.md` §25):

| regime | faster | margin |
|---|---|---|
| degree 2, any rank (16 → 32) | ray | 1.5–2.1× |
| degree ≥ 3, rank ≥ 10 | peel | 1.1–1.7× |
| rank ≤ 4 | peel | up to 3.9× |
| wherever the monomial-charge gate trips | **only the factor engine builds at all** | — |

An earlier reading of this repo's own benchmark as "~3× faster at rank 12–24" was
an artifact of that case mix: its high-rank entries were all at degree 2, so the
figure measured *degree*, not rank.  Retracted; the grid above supersedes it.
**Coverage, not speed, is what the factor engine is for.**

Bulk check of the swap
(`experiments/s_ray_spectrum_dictionary_sweep.py`): over the canonical
BPS-quiver dictionary, **1108/1108** entries at ranks 1–8 give ray `==`
`S_from_spec` (ground truth, since `S` is chamber-independent) `==` peel.  (The
leading-data count is a *regression guard*, not corroboration — it is satisfied by
construction for **any** leading data, measured; `repo_audit.md` A44.)

**The factor engine's own cost is set by the ORDER more than by anything else, and
badly enough to change the cost class** (`s_leading_term_existence.md` §26,
`experiments/s_ray_order_heuristics.py`).  Cost tracks the number of rays placed =
the number of BPS states in the chamber the chosen order selects.  On a quiver
with an infinite chamber the good order stays at `rank` rays **at every cutoff**
while a bad one pays for the whole tower: Kronecker-3, good against reversed, is 2
rays against 13/20/29/40 at `D = 6/8/10/12`, a time ratio rising 13.5× →
**118.6×**.  On finite-chamber quivers it is a bounded 2–4×.

The good order is **free to compute where the quiver is acyclic**, it is this
repo's own **mantle theorem** (acyclic ⇒ topological product spec), and **it is now
the default** — `order="strip"`: `acyclic_node_order` for the node order,
`mean_rank_key` to extend it to the cone.  **No central charge is involved**
(user, 2026-08-13); a mean has the one property — a sum sits between its summands —
that a linear `Z` was contributing, and it reaches the same minimum-ray chamber
exactly (`rank` rays, spin-0 only).  Where the strip leaves a core that is **not
one strongly connected component**, the default is the **component order**
`order="component"` (user, 2026-09-23: *"The S-builder can also be improved along
the same lines, effectively picking a total order compatible with the
decomposition"*): the strongly connected components in source-first order, each
built on its own cone in its own default order, and `S` assembled as their ordered
product — in that order `Ω = 0` is forced on every charge meeting two components
(`strongly_connected_redesign.md` §1.3), and each coefficient of the product is a
single term, since a charge splits among the components in one way only.
Measured 2.4–6.2× faster than `"random"` on quivers of two or three cycles, the
gap growing with the cutoff, with the same `S` and fewer factors.  Only on a
strongly connected quiver, where no order is determined, does the default fall
back to `order="random"`: each `(s, γ)` piece at its own position, which `S` is
measured to be invariant under (`experiments/s_ray_split_spin_pieces.py`).  A
central charge is used **only** when `phases=` is supplied — see
`s_leading_term_existence.md` §27.

**The gate bites on catalogued theories, and only at sufficient depth.**  At
cutoff 3–4 the peel engine builds every dictionary entry; at **cutoff 5** it
honest-fails on the `circular_2_U2-U2` family (variants at ranks 4, 5, 6) — a
circular U(2)×U(2) quiver, the matter-carrying case the monomial-charge gate exists
to catch — and the factor engine builds those four and matches ground truth on them.
The same cutoff-sensitivity is on record for SU(3)+N_f=1 (passes at cutoff 4,
fails at 5; `recursive_spectrum_generator.md`), so a coverage comparison taken at
one shallow cutoff understates the gap.

**What the swap does and does not buy at Markov / N=2\*.**  It moves the frontier
from *cannot build `S` at all* to *`S` builds and `multiply` works* — and no
further, which is worth stating precisely because it is easy to over-read.  σ/ρ
do **not** close there: `upper(F_{γ_a})` was measured landing exactly on the cone
boundary at every cutoff 6…10, so this is not a matter of raising the cutoff, and
the natural reading is that `F_{γ_a}`'s cone support is genuinely unbounded — which is
what having no finite `E_𝖖`-product would predict.  `multiply` at Markov is
therefore cone-truncated to the built degree, exactly as spec-free mode
documents.  Getting σ/ρ at a theory with no finite chamber is a separate open
problem, not an engine choice.

Swept across the whole BPS-quiver dictionary
(`experiments/s_ray_stress_dictionary.py`, 2026-08-13): on **all 1108 entries**
of `n_001 … n_008` (rank ≤ 8) the factor build equals the entry's own spec-derived
`S` at **every** cone degree, the three placement orders give the same element,
and `S` at cutoff `D` restricted below `D` equals `S` at cutoff `D−1`.  Since the
BPS factorisation is unique per order (injectivity — proved), that equality says
each entry's physical chamber **is** the BPS factorisation in its own order.  The
same holds with *arbitrary* integer leading data, where no oracle applies but
order-independence still must.

Two things about the factor engine that are easy to get wrong:

* **`S` is order-independent; the factorisation is not.**  Where the BPS factors
  are placed changes `Ω` — the ray *content* — but not the element.  So `Ω` is a
  chamber's spectrum only in the physical (central-charge phase) order, and the
  **central charge chooses the chamber**: a generic one puts pure SU(2) in its
  strong-coupling chamber (2 rays), the weak-coupling one gives the dyon tower
  plus the W boson (`Ω = 𝖖⁻¹+𝖖`).  Same `S`.
* **Never `𝖖`-truncate it.**  Cone truncation is the only sound one; negative
  brackets shift high-`𝖖` terms down into the visible window
  (`s_leading_term_existence.md` §5).

### F-finding

* **(a) Direct solve at root** -- `BPSKAlgebra._F_internal(γ)` calls
  `solve_F_via_s_coefficient` over the doubly-tropical interval
  `[γ, ν_S(γ)]` in the chart's positive cone.
* **(c) Chart-graph chain** -- `BPSKAlgebra._solve_F_via_chain(γ)`:
  greedy descent through the chart graph picking forward mutations
  that minimize `L1_in_cone(u − l, dst.cone)`, solve at chart-end,
  inverse-transport back.  Falls back to (a) if no chain helps.

### Multiplication

* **Direct at root** -- `BPSKAlgebra.multiply(a, b)` (delegates to
  the chart's `bps.CoulombAlgebra.multiply` greedy peel).
* **Chart-search multiply** -- `BPSKAlgebra._multiply_via_chart_search(a, b)`:
  greedy descent minimizing `|F(L_a) in chart| × |F(L_b) in chart|`,
  multiply at chart-end via a fresh `BPSKAlgebra`, translate
  decomposition labels back to root via inverse μ chain.

Both algorithms apply spec-tightening at the chart-end before solving.

* **(d) Joint with `S`, no `S` input** -- `fs_builder.FSBuilder` (`build_F_and_S`), the
  crystalline route (user direction 2026-08-13;
  `pronilpotent_group_conjecture.md` §9).  Where (a)–(c) all take `S` as given and
  enumerate a support window, this grows `F_γ` and `S` **together** out of
  `F·S = X_γ + O(𝖖)` plus `S`'s leading data: at each cone degree it places the
  BPS factors that the leading data forces and then reads off the palindromic
  `F`-coefficients the relation forces, degree by degree.  So no spec, no `S`, and
  **no `σ⁻¹`** — the doubly-tropical interval is an *output* here rather than the
  enumeration window, which is what makes agreement with (a) evidence.  The two
  moves are one alphabet: `[n]_𝖖 = χ_{(n−1)/2}`, the F-solver's peel basis and the
  factor builder's multiplicity basis being the same `Z`-basis of the palindromic
  Laurent polynomials.  ⚠ The relation *alone* pins nothing (`Ω ≡ 0` gives
  `S = 1`, `F = X_γ`, which satisfies it) — the leading data is what makes the
  build deterministic, and `omega_policy=` exposes the remaining choice.
  **Why the two recursions interleave** (user, same day): adjoining a node at
  `γ + γ_0` with `γ_0` pure flavour gives
  `S_new = S − 𝖖/(1−𝖖²)·X_{γ_0}·F_γ S + O(X_{2γ_0})`, so `F_γ S` is the
  `X_{γ_0}`-linear sector of an ordinary `S` build and (d) is that build
  linearised — one construction, not two.  It follows that **`F_γ·S` is a
  presentation, not the object**: the sector is generally a sum of terms with the
  `X` monomials inserted at various positions between the `E_𝖖` factors, and
  `F_γ` on the far left is the leftmost placement.  `F_from_enlarged_quiver`
  builds `F_γ` that way through `BPSFactorSpectrum` alone, giving a three-way check
  against (d) and (a); `fs_linear_sector` places each `(s, γ')` piece of either
  kind individually (`BPSFactorSpectrum(piece_key=…)`), so the summands may sit
  anywhere among the `E_𝖖` factors — the element is measured invariant under
  that (480/480, six rules), only its split into summands is not.

### Strategy (b) `S_1 F' S_2 = X_γ + O(q)` -- enrichment of (c)

`BPSKAlgebra._solve_F_via_chain` runs the (c) descent and then, at
chart-end, optionally takes a *(b)-shortcut* when the chain has
preserved the `(S_2, -S_1)` boundary structure.

Boundary tracking lives on `Chart`:

* `s1_chain` -- the consumed-head charges along the chain so far.
* `boundary_preserved: bool` -- True iff every chain step was a
  forward (or inverse) necklace at the actual head/tail with no
  local moves applied.  Local moves can rearrange the spec across
  the `s2_length`-th boundary, destroying preservation.
* `s2_length = len(spec) - len(s1_chain)` -- when preserved, the
  index of the first `-g_i` factor in the chart's spec.

When boundary is preserved at chart-end and `s1_chain` is non-empty,
`_try_solve_F_via_b` solves `S_1 F' S_2 = X_γ + O(q)` for `F'` (with
F'-support enumerated in chart-end's cone) and inverse-transports
across `S_1` to recover root-F.  A round-trip check
(`_lm_solve(F_recovered, S_1) == F'`) catches cases where the
modified solver produced an algebraically-invalid F' and falls
through to standard (c) (local solve + transport-back).

Infrastructure: `bps_kalgebra_internals.solve_F_modified` and
`s1_xdelta_s2_coeff` (triple Nahm sum).  The earlier standalone
`(S_1, S_2)`-split search (`transport_search.py`,
`find_split_minimizing_F_support`, `predict_F_support_evolution`)
was removed once the in-chain (b)-shortcut superseded it.

## Mutation rules (cluster necklacing)

For a forward mutation at the head `g` of the spec:

| object | rule |
|---|---|
| spec | `spec → [spec[1], …, spec[N-1], -g]` (rotation only — *no* μ on intermediate factors) |
| nodes | FZ: node `g` flips to `-g`; others `γ_j → γ_j + max(⟨γ_j, g⟩, 0) · g` |
| lower trop `l` | `l → μ_g(l) = l + max(⟨l, g⟩, 0) · g` |
| upper trop `u` | `u → ν_g(u) = u + max(⟨g, u⟩, 0) · u` |

Inverse mutation reverses each rule consistently.  These three
different rules (spec rotation, FZ on nodes, μ vs ν on bounds) are
the cluster-mutation framework's documented update laws.

## Isomorphism

`bpskalgebra_iso.find_isomorphism(alg_1, alg_2)` returns a
*sufficient* witness `Iso(A, local_move_chain)`:

* `A`: unimodular integer matrix with `A^T B_2 A = B_1` (intertwines
  pairings) and `A · alg_1.nodes = alg_2.nodes` as multisets.
* `local_move_chain`: pentagon collapses + commute swaps + pentagon
  expansions transforming `[A · g for g in alg_1.spec]` into
  `alg_2.spec`.

When `find_isomorphism` returns `None`, we have **not proven** the
algebras are non-isomorphic.  The witness encodes one provable
sufficient condition; absence of a witness leaves the question open
(potentially distinct algebras with the same quiver may correspond to
different superpotentials).

`find_weak_isomorphism(alg_1, alg_2, max_depth_1, max_depth_2)`
extends to "any chart pair" -- enumerates charts in both algebras'
chart graphs up to budget, and tries strong-iso for each pair.

**`KAlgebraObject` (`kalgebra_object.py`, Plan 25)** holds *one
abstract algebra* as its realizations plus the `KAlgebraIso`
witnesses between them — a connected groupoid with composed
transport (`iso(src, dst)` / `transport(label, src, dst)`),
capability routing (`preferred('trace-exact')`), the per-witness
`verify_all` battery (`verify_pairwise`), and a path-independence
certificate (`verify_coherence`; isos between two presentations form
a torsor under `Aut`, so coherence is a property of the *curated*
witness set, not automatic).  Deliberately **not** itself a
`KAlgebra` (decisions D3: label-space ambiguity breeds
presentation/object category errors; an `as_kalgebra(authority)`
view can be added if a consumer needs it).  First populations:
`finite_kalgebras.objects.kalgebra_object('pentagon' | 'heptagon'
| …)` — pentagon carries four certified presentations (frozen cone,
BPS chart, `PentagonKAlg`, `A1A2kKAlg(1)`), with the cone↔BPS
witness canonical (labels are `γ = Σ p·γ_i` sums; inverse by exact
ray decomposition) and the cone↔cone dictionaries found by
`match_generators` (ρ-orbit assignment search certified by the full
battery, including trace-equivariance of the chiral characters).

**`RGKAlgebraObject` (`rgkalgebra_object.py`)** refines the holder one
level: it identifies *flows*, not just algebras.  A witness is an
`RGKAlgebraIso(KAlgebraIso)` — the UV-side iso plus an `aux_iso` of
the auxiliaries — with flow verifiers (`verify_rg_intertwine`,
`verify_s_rg_match`, `verify_apex_match`); `aux_object()` projects to
the `KAlgebraObject` of the auxiliaries.  The distinction is
executable: a chart-mutation witness (`bps_chart_object`) certifies
the *algebras* equal but fails `verify_s_rg_match` (different chamber
= different flow), while a unimodular frame change
(`bps_frame_change`) passes the full flow battery.

## Conventions and external dependencies

`BPSKAlgebra` and the chart graph use:
* `lattice_torus.Lattice` for the pairing.
* `lattice_mutation.{solve, solve_inverse}` for algebra-level transport
  across single E_q factors.
* `lattice_canonical.{sigma_forward, sigma_inverse}` for tropical
  σ / σ⁻¹ (chart-internal).
* `nahm_data.s_gamma_habiro` for `[S|0⟩]_γ`.
* `bps_quiver_tools.CoulombAlgebra` as the canonical-basis decomposer
  (private state in `BPSKAlgebra._chart`).

## Parity contract — `TKAlgebra`, `OKModule` and their realisations (Plan 06)

Plan 06 adds a parity layer on top of the `KAlgebra` contract: an
anti-involution `τ` on the canonical basis (time-reversal /
*-operation), the composite parity `τρ`, an OKModule notion of left
A-modules with their own canonical basis, and the BPS-realisation
machinery that makes both concrete on a palindromic chamber.

### `TKAlgebra(KAlgebra)` — parity-extended algebras

A `TKAlgebra` is a `KAlgebra` together with two new abstract
primitives:

```python
class TKAlgebra(KAlgebra):
    @abstractmethod def tau(self, a: Label) -> Label
    @abstractmethod def tau_R_basis(self, b)             # involution on R
```

with derived `parity = τρ`, `tau_element` (Z-form, twist-free),
`tau_R_element` / `tau_R_laurent` / `tau_R_form` (R-side / R-form
view).  Mathematical content (T1–T5):

* (T1) `τ : Labels → Labels`        permutes the basis
* (T2) `τ²(a) = a`                    involution
* (T3) `τ(1) = 1`                     fixes the identity
* (T4) `τ(L_a · L_b) = L_τ(b) · L_τ(a)`   anti-homomorphism
* (T5) `τρτρ = id`                    parity is an involution

Plus the `τ_R` axioms on `R`: (TR1) involution, (TR2) fixes `1_R`,
(TR3) commutes with the Z₊-ring `⋆`.  At the Z-form level
`tau_element` carries `LaurentPoly` coefficients through unchanged
(`q` is fixed by τ); `τ_R` only surfaces in the R-form view via
`tau_R_form`.

Verifiers (full bundle in `verify_parity_axioms`):
`verify_tau_is_involution`, `verify_tau_fixes_identity`,
`verify_parity_is_involution`, `verify_tau_is_anti_homomorphism`,
`verify_parity_is_anti_homomorphism`,
`verify_tau_R_is_involution_basis`,
`verify_tau_R_fixes_one_basis`,
`verify_tau_R_commutes_with_star_basis`.

### `TRGKAlgebra(RGKAlgebra, TKAlgebra)` — RG / parity bridge

Adds `verify_rg_tau_equivariant(a)` (`RG(τ_UV(a)) == τ_IR(RG(a))`)
and `verify_rg_generator_is_parity_invariant(K)`.  A second variant
`TTRGKAlgebra` exists for theories where the τ-twist enters only on
the auxiliary side.

### `TBPSKAlgebra(BPSKAlgebra, TRGKAlgebra)` — BPS realisation

Constructor takes `tau_IR: GL(rk Γ, Z)` satisfying
`⟨τ_IR γ, τ_IR γ'⟩ = -⟨γ, γ'⟩` and palindromic-spec compatibility.
Label-level τ is derived from `τ_IR` and `σ` by

    τ(a) = -τ_IR(σ⁻¹(a))

This σ-formula is currently a **conjecture** — empirically verified
on 74 cases against the operational definition
`F(L_{τ(a)}) = τ_IR(F(L_a))`, all matching, but no proof is in.  When
implementing new theories, build with `tau_IR=...` and run the
`verify_*` bundle; mismatches with the operational definition are
the most likely first sign of a bug (see Plan 06 contract notes).
*Further evidence, 2026-09-08 (Plan 06 `decisions.md`, the
parity-classification entry):* at pure SU(2) the operational definition
holds 46/46 for the Kronecker chart's `τ_IR` and the σ-formula reproduces
it 46/46, while the other pairing-flipping involution `diag(1,−1)`
satisfies the operational definition on 4/46 labels only — so the formula
is being tested against a genuine alternative, not vacuously; at pure
SU(3) both arrow-reversing involutions of the strong-coupling chart pass
81/81 on `[−1,1]⁴`, including the cross-pair one, whose `S` is
`T`-invariant but not list-palindromic and therefore never enters this
constructor (`experiments/parity_bps_dictionary.py` [retired 2026-09-19 with the type-A keystone; the measurement stands as recorded]).  The operational
definition is what establishes a parity at the algebra level when the
palindromic-spec gate refuses the spec.

### `OKModule` — orbi K-modules with canonical basis

An `OKModule` over a `TKAlgebra A_𝖖` is a left A_𝖖-module **defined
over `Z[q, q⁻¹]`** with a distinguished canonical basis `{M_s}`.
Z-form `ModuleElement = dict[ModuleLabel, LaurentPoly]` (no R / OR
involvement at the canonical-basis level — that lives in the orbi
inner product output, mirrored on `Element` vs `ElementOverR` for
the algebra).

Six abstract primitives:

```python
class OKModule(ABC):
    @abstractmethod def algebra(self) -> TKAlgebra
    @abstractmethod def coefficient_ring(self) -> ZPlusRing      # OR = module-side ring (R^{τρ} for τρ-fixed flavour)
    @abstractmethod def module_basis_iter(self) -> Iterator[ModuleLabel]
    @abstractmethod def module_action(self, a: Label, s: ModuleLabel) -> ModuleElement
    @abstractmethod def orbi_inner_product(self, r, s, K) -> RPowerSeries
    @abstractmethod def _module_label_section_decompose(label) -> ...
```

Coefficient ring `OR` (= module-side ring; for τρ-fixed-flavour
content `OR = R^{τρ}` is the τρ-fixed subring of the algebra `R`).

Defining axioms:

* (M1) Canonical basis `{M_s}` over `Z[q, q⁻¹]`.
* (M2) Module action `c^t_{a,s}(q) ∈ Z[q^±]` and (M2') associativity.
* (M3) `τρ`-mirror: `c^t_{a,r}(q⁻¹) = c^t_{τρ(a),r}(q)`.
* (M5) Orbi inner product `OI_{r,s}(q) ∈ OR((q))`.
* (M6) τ-adjoint: `(L_a · M_r, M_s) = (M_r, L_{τ(a)} · M_s)`.
* (M7) Orthonormality: `OI_{r,s} = δ_{r,s} + O(q)`.

Verifiers `verify_orthonormality` (M7), `verify_identity_action`
(M2 at `a = 1`), `verify_associativity` (M2'), `verify_tau_adjoint`
(M6), `verify_module_mirror` (M3), bundled in
`verify_okmodule_axioms`.

### `QTTauModule(OKModule)` — canonical QT-side OKModule

Over `TauQuantumTorusKAlg(B, τ_IR)`.  Y-state basis labelled by
`Λ := ker(I + τ_IR)` (the τ_IR-anti-fixed sublattice, Lagrangian
for `B`); magic-torsor-twisted shift action

    X_γ · Y_δ = q^{−½⟨γ, τ_IRγ⟩ + ⟨γ, δ⟩ + ν(γ)} · Y_{δ + γ − τ_IRγ}

with quadratic refinement `ν : Γ → ½Z` solved subject to (i)
integrality at each lattice basis direction
(`ν(e_i) ≡ ½⟨e_i, τ_IR e_i⟩ mod Z`) and (ii) `ν|Λ = 0` (magic-torsor
covariance).  Bare diagonal pairing `(Y_r, Y_s) = δ_{r,s}` is the
canonical OI on the QT side.

### `BPSOKModule(OKModule)` — BPS-realisation OKModule

Mirrors the algebra-side `BPSKAlgebra`:

| algebra side                   | module side                            |
|--------------------------------|----------------------------------------|
| `QuantumTorusKAlg`             | `QTTauModule`                          |
| `BPSKAlgebra(RGKAlgebra)`      | `BPSOKModule`                          |
| `F_γ · S \|0⟩ = X_γ \|0⟩ + O(q)` | `S_+ · V_s = Y_s + O(q)`              |
| F-solver lives in `BPSKAlgebra`| V-solver lives in `BPSOKModule`        |

Implementation:

* **V-solver**: BFS palindromic completion analogous to `solve_F`.
  Walks Λ-labels in cone-monotone order from `s`; at each `t`,
  computes the partial `S_+ V_s` coefficient at `Y_t`, takes its
  non-positive q-part, palindromises, and adds the correction.
* **`S_+` coefficients**: computed exactly via Habiro/Nahm sums in
  `_s_plus_y_habiro_table` (one `HabiroElement` per target Λ-label).
  No explicit multiplication of E_q factors.
* **`G_a` for the module action**: `F(L_a)` halfway-commuted through
  `S_+` via `lattice_mutation.solve` (intertwining
  `F(L_a) · S_+ = S_+ · G_a`).  `module_action` uses `G_a`.
* **OI**: bare-Y diagonal pairing of `S_+ V_r` against `S_+ V_s` —
  no infinite-product prefactor.  That is a **convention** of the
  module pairing, *not* a consequence of the magic torsor: `ν`
  enters the action only as a monomial exponent (`module_action`
  returns `q^{q_exp}·M_{…}`), and a monomial cannot cancel
  `(𝖖²;𝖖²)_∞^{rk}`.  What `ν|Λ = 0` buys is that the bare-`Y`
  **diagonal** carries no `q`-shift; the QT-side
  `orbi_inner_product` then returns `δ_{r,s}` by construction.
  (Corrected 2026-09-05 — the earlier wording "the magic-torsor
  convention removes it" asserted a derivation that does not
  exist, and was then used to explain the M6 residual, which it
  cannot: `repo_audit.md` A71.)

**Scope of the current PR.** Even-palindromic spec, no `T`-fixed
factor (`T = τ_IR ∘ ρ_IR`), and `h := |spec_half| ≤ rk Λ`.
Pentagon-palindromic, pure-SU(2)-palindromic, and 14/19 buildable
palindromic-dict entries pass `verify_tau_adjoint` (M6).  Two
follow-up axes are flagged but not in scope:

1. ~~**τ_IR-invariance of ν**~~ — **RETRACTED 2026-09-04.**  This said the
   ν solver enforces integrality + `ν|Λ = 0` only, that for some theories the
   resulting ν fails `ν(τ_IR γ) = ν(γ)` and breaks M6, and that the fix is a
   third linear constraint.  All of it is wrong: ν is **Q-linear** and
   `im(τ_IR − I)` spans the same rational subspace as `Λ`, so `ν|Λ = 0`
   **implies** τ_IR-invariance — the solver's own note (b) ⟺ (c) says so.
   Measured: `τ_IR^T ν = ν` on all 8 modules built from dictionary entries,
   including the 6 that fail M6.  The proposed third constraint is a no-op and
   **M6's residual is not a ν problem**.  What ν *does* control is WHICH
   orbi-module you get, through `nu_on_flavour_basis` on τ-**fixed** flavour —
   now reachable from `BPSOKModule.__init__`, and measured to move the
   orbi-index while leaving M6 unchanged.  M6's actual cause is open.
2. **Kernel direction `h > rk Λ`**: the n-walk has a kernel
   direction producing infinite Nahm tuples per target with
   shift = 0 — V_s defined by an algebraic identity rather than a
   finite cone walk.  No physically motivated `σ`-analogue exists
   for the V-solver, so halting requires either a Theta-series
   identification or a growth-bound conjecture.

### `nu_on_flavour_basis` family parameter

For τρ-fixed-flavour content `Γ_f ∩ Γ^{τ_IR}`, the OKModule has a
family of presentations parameterised by `ν|Γ_f^{anti-τ}` (the
linear piece of the quadratic refinement on the τ-anti-fixed
flavour basis).  `qttau_module._compute_quadratic_refinement` accepts
a `nu_on_flavour_basis` kwarg.  Plumbing this through
`BPSOKModule.__init__` is a future cleanup.

## Pending / future work

* **Gauge-lattice projection helper** for theories outside the
  dictionary (the canonical dictionary at `dictionaries/` already has
  gauge-projected entries).
* **Tuning algorithm-(c)'s descent heuristic** ("keep searching
  charts vs. solve here") using bench data.
* **Multi-factor non-abelian flavour** (current rings cover a single
  SO(3) / SU(3) factor; see "Flavour in `KAlgebra`").
  <!-- Migration to the repo root: DONE (Plan 07, 2026-05-10). -->
* **OKModule kernel-direction case** (h > rk Λ): Theta-series
  identification of `V_s` for entries where the Nahm walk fails to
  halt (Plan 06 follow-up).
* ~~**τ_IR-invariance of ν**~~ — **RETRACTED 2026-09-04** (see the Plan 06
  scope note above): `ν|Λ = 0` already implies it, so the proposed third
  constraint changes nothing and cannot lift M6.  M6's cause is open.
* **σ-conjecture proof / promotion**: settle whether
  `τ(a) = -τ_IR(σ⁻¹(a))` is a theorem or define `τ` operationally
  via `F(L_{τ(a)}) = τ_IR(F(L_a))` directly.

## Cross-reference with the `K_𝖖-algebras` draft (user, 2026-09-19; refreshed against the 2026-09-21 version)

The user's draft paper *`K_𝖖`-algebras* (D. Gaiotto; the companion of this
repository, cited by its LaTeX labels since the draft itself is not tracked
here) is *"by now sufficiently stable that we can cross-reference it with the
repo"* (user, 2026-09-19).  On 2026-09-21 the user supplied the current source
(`nice_temp.tex`, amsart, preamble `preamble_ams.tex`) so that *"the
documentation of the repo refers to it correctly, definitions, conjectures,
remarks, examples"*; this section was re-derived from that version, label by
label.  This table maps the draft's definitions, remarks, examples, equations
and conjectures to the repo objects that encode or test them.  Labels are the
draft's; repo axioms are cited by name as always.  The reader-facing
counterpart of this table — organised chapter by chapter like the paper, one
implement / generate / test / standing block per structure — is
`paper_companion.tex` at the repo root, marked on its title page, in its
abstract and in a front Statement of AI use as an AI-written extension of the
draft (user, 2026-09-21: *"mark the companion clearly as an AI-written
extension of my draft … Proper attribution of AI use is key to a developing
ethical framework"*) (user, 2026-09-21: *"a document sibling
to the draft which indicates which repo machinery can be used to implement,
generate and test the various mathematical structures. A paper companion"*);
it compiles beside the draft with the draft's own preamble and numbers, or
standalone with a fallback preamble.

**What changed between the 2026-09-19 pass and the 2026-09-21 version** (each
fixed below): the trace-equivariance axiom is no longer "being added" — it is
K5 with label `ax:rhotr`, and its flavoured form is F5 of the flavoured
definition, whose forgetful map moved to **F6**; the RG-flow pairing axiom is
relabelled **`ax:rgtrace`** (it was `ax:trace`, a label the draft now keeps for
K4 only); wall-crossing has its own label **`def:wallcrossing`**; and about
fifteen remarks, definitions and equations are new or were never mapped
(the maximal-extension remark, the Webster comparison, the three "miracle"
remarks, the flavour-equivariance remark, the `U_𝖖(sl_2)` trace recursion,
the composition and `ρ⁻¹`-family remarks on RG flows, the subgroup `𝓔` and
its `g⁻¹(𝖖) = g(𝖖⁻¹)` remark, the flavoured dilogarithm `eq:fleq`, the two
interface indices of Appendix A, the finite-type table).

**Printed numbers, 2026-09-21 version** (a dated snapshot — the shared
theorem counter renumbers whenever a remark is inserted, so cite the LABEL and
use these only to find the item in the PDF): Definition 2.1 `def:kq` ·
Remarks 2.2 (trace rescaling), 2.3 (Webster) · Example 2.4 `ex:qt` · Remark
2.5 `rem:maximal` · Definition 2.6 `def:penta` · Remarks 2.7 (existence), 2.8
`rem:miracle` · Definition 2.9 `def:kq-flavoured` · Remark 2.10
`rem:flavour-change` · Example 2.11 `ex:qt-flavoured` · Remark 2.12
`rem:flavour-comments` · Definition 2.13 (`U_𝖖(sl_2)`, unlabelled) · Remark
2.14 (ADE variants) · Definition 2.15 (punctured triangle, unlabelled) ·
Remarks 2.16, 2.17 `rem:a1d3-miracle` · **Conjecture 1** `conj:abeKalgebra` ·
Definition 4.1 `def:rg` · Remarks 4.2 (reconstruction), 4.3 (composition), 4.4
(`ρ⁻¹` family) · Definition 4.5 `def:wallcrossing` · Remarks 4.6 (central
charge), 4.7 (matter removal) · Definition 4.8 (`𝓔`, unlabelled) · Remark 4.9
(`g⁻¹(𝖖) = g(𝖖⁻¹)`) · Definition 4.10 `def:sw` · **Conjectures 2–3**
`conj:S-unique`, `conj:pbw` · Remarks 4.11, 4.12 · **Conjectures 4–8**
`conj:cluster`, `conj:dt`, `conj:coha`, `conj:upper`, `conj:factored` ·
Definition 4.13 (`𝓔_{G_f}`, unlabelled) · **Conjecture 9** `conj:junctions` ·
Remarks 5.1, 5.2 · Appendix A `app:physics` · Appendix B `app:finite`,
`app:a1a2k`, Table B.1 `tab:finite_type`.  Conjectures carry their own global
counter.  (Numbers confirmed against the compiled PDF of that version.)

⚠ **One label is defined twice in the draft:** `\label{ax:rhotr}` is attached
both to K5 of `def:kq` and to F5 of `def:kq-flavoured`, so a `\ref{ax:rhotr}`
resolves to F5.  Below, "K5" and "F5" are written out rather than cited by that
label.

| draft | statement (draft's conventions) | repo |
|---|---|---|
| `eq:intro-delta`, and the intro's companion-repository footnote | `I_{a,b} = Tr L_{ρ(a)}L_b = Tr L_b L_{ρ⁻¹(a)} = δ_{a,b} + O(𝖖)`; the public GitHub companion | `inner_product`, `verify_orthonormality`; the public tree is staged in `export/public/KAlgebra/` (`export/README.md`) |
| Def. `def:kq` K1 (`ax:bar`) | antimultiplicative bar involution, `𝖖 ↦ 𝖖⁻¹` | `Element.bar`, `verify_bar_involution` |
| K2 (`ax:basis`) | bar-invariant canonical basis containing `1` | labels + `verify_identity_in_basis`; bar-invariance of every built element is W1 on the Abe tier (`certify_canonical`) |
| K3 (`ax:rho`) | automorphism `ρ` permuting the basis | `rho`, `rho_inverse`, `verify_rho_is_automorphism`, `verify_rho_fixes_identity`, `verify_rho_inverse` |
| K4 (`ax:trace`, `eq:orthonormality`) | `ρ²`-twisted trace, `I_{a,b} = Tr L_{ρ(a)}L_b = Tr L_b L_{ρ⁻¹(a)} = δ_{a,b} + O(𝖖)` | `trace`, `inner_product`, `verify_rho_twisted_trace`, `verify_trace_pairing_faces` (the two faces), `verify_orthonormality` |
| K5 (label `ax:rhotr`, shared with F5) | `Tr L_{ρ(a)} = Tr L_a`, implying `I_{b,a} = I_{a,b}` | axiom 5, **ρ-equivariance of the trace** (this file, "Axiom 5"): `verify_trace_intertwines_rho` (the unflavoured statement, `⋆ = id`), `verify_trace_intertwines_rho_star`, `verify_pairing_rho_star_symmetric`; inheritance table there; manifest on the Abe tier from the closed-form sector measure |
| Remark 2.2 (first remark after `def:kq`) | the trace may be rescaled by `1 + O(𝖖)`; physics fixes a normalisation | the `(𝖖²;𝖖²)_∞^{2·rk}/|W|` prefactor of `trace_residual` — the draft's `(𝖖²;𝖖²)_∞^{2 rk 𝔤}/|W|` of `eq:measure` |
| Remark 2.3 (second remark after `def:kq`) | the canonical basis and the `δ_{a,b}` axiom play a similar role in Webster's work, with `𝖖_here = q⁻¹_there` | no repo counterpart; the relevant discipline is ONBOARDING.md "False friends" (the bar involution here is antimultiplicative, the basis is not assumed to be any published one) |
| Ex. `ex:qt` | quantum torus, `Tr X_γ = (𝖖²)_∞^{rk} δ_{γ,0}`, `ρ(γ) = −γ`, `ρ² = id` | `QuantumTorusKAlg` (`BPSKAlgebra`'s IR auxiliary); `TauQuantumTorusKAlg` |
| Rem. `rem:maximal` | the `δ_{a,b}` axiom is expected only after the collection of line defects is maximally extended, direct summands included | no verifier: each presentation's canonical basis is its full label set by construction.  This is NOT the line-lattice maximality of `sec:coulomb` (below), which `global_form.LineLattice.verify_maximal` does check |
| `sec:pentagon`, Def. `def:penta`, `eq:izero`, `eq:pentarec`, `eq:ABdef`, Table `tab:pentagon-stabilization` | the pentagon algebra `K_𝖖([A_1,A_2])`: relations, `L_{i;a,b} = 𝖖^{ab}L_i^aL_{i+1}^b`, `ρ(L_i) = L_{i+2}`, the trace seeds, the recursion `Tr L_i^n = 𝖖^{1−2n}Tr L_i^{n−1} + 𝖖^{2−2n}Tr L_i^{n−2}`, `𝖖^{n²−1}Tr L_i^n = A_n Tr 1 + B_n Tr L_1` | `pentagon_algebra.py` + `implementations/pentagon_trace.py` (`t_coeffs(n)`; **the table's four rows `n = 3..6` reproduced exactly, 2026-09-21**), `pentagon_cone_data.py`, `finite_pentagon_kalg.py`, `PentagonKAlg` (`kalgebra_samples.py`); as a BPS chart `BPSKAlgebra(pairing=[[0,1],[−1,0]], …)` — whose `I_{(1,0),(1,0)} = 1 − 𝖖² + 𝖖⁴ + 𝖖⁶ + O(𝖖⁷)` is `Tr 1 + 𝖖 Tr L_i` from the draft's seeds; `finite_type_kalgebras.md` |
| Remark 2.7 (after `def:penta`) | existence follows from the cluster presentation (Gaiotto–Moore–Neitzke) plus the IR Schur-index formulae of Córdova–Gaiotto–Shao, formalised as a Seiberg–Witten flow | exactly what the BPS chart is: `BPSKAlgebra` builds the pentagon from its quiver and `verify_*` checks the axioms; the draft's `\ref{def:sw}` here is to a Definition (it prints as "Section 4.10") |
| Rem. `rem:miracle`, Rem. `rem:a1d3-miracle`, App. B "The minors miracle" | `A_∞ = −Tr L_1`, `B_∞ = Tr 1`; the `2×3` minors reproduce `Tr 1, Tr T_0, Tr D_0`; `(−1)^s det 𝓜^{(ŝ)} = T_s` for the odd polygons | Goal 2.11 "the trace-miracle puzzle" (`RESEARCH_GOALS.md`), `pentagon_miracle_reduction.md`, `restructuring_plans/34_miracle_qdifference_closure/`, `a1d3_trace_recursions.md`, `a1a2k_kq_proof.md` (the trace program) |
| Def. `def:kq-flavoured` F1–F2 (`ax:f-bar`, `ax:f-basis`) | flavoured bar / basis, free over `Z[𝖖^±]⊗R_{G_f}` non-canonically, ambiguity = 1d irreps | Plan 32: `r_label_decompose` (`L_label = χ_w·L_section`), `verify_section_is_single_irrep`, "sections single-irrep mod Λ" |
| F3 (`ax:f-rho`) | `ρ` twisted-linear over `R_{G_f}`, `ρ(χ_r) = χ_{r^∨}` | `⋆` on coefficients (`RPowerSeries.star`); the ⋆-semilinear `ConeKAlgebra.rho_element`; `verify_embed_intertwines_rho` |
| F4 (`ax:f-trace`, `eq:f-index`, `eq:f-integrality`, `eq:f-orthonormality`) | `I_{a,b} ∈ R_{G_f} + 𝖖R_{G_f}[[𝖖]]`, identity summand `δ_{a,b} + O(𝖖)` | `to_R_form`, `verify_inner_product_consistent`, `verify_orthonormality` on the R-form; integrality of indices is a DIAGNOSTIC on the interface tier (`interface_indices.md`) |
| F5 (label `ax:rhotr`, shared with K5) | `Tr L_{ρ(a)} = ρ(Tr L_a)`, implying `I_{b,a} = ρ(I_{a,b})`, *"the outer `ρ` just acts on the flavour ring"* | axiom 5's flavoured form: the outer `ρ` is `⋆` (`RPowerSeries.star`, `χ_r ↦ χ_{r^∨}` by F3); `verify_trace_intertwines_rho_star`, `verify_pairing_rho_star_symmetric`; `RingHom.verify_commutes_with_star` is what lets it transport through `base_change` |
| **F6** (`ax:f-forget`), Rem. `rem:flavour-change` | forgetful map `χ_r ↦ dim r` commuting with `ρ` and `Tr`; lowering along `H_f → G_f` | `KAlgebra.forget()`, `lower_flavour(φ)`, `base_change(RingHom)`, `add_flavour(R)` (`flavoured_kalgebra.py`), `augmentation_hom` / `restriction_hom` |
| Ex. `ex:qt-flavoured`, `eq:qt-flavoured-trace` | flavoured quantum torus, kernel `Γ_f`, `Tr X_γ = (𝖖²)_∞^{rk(Γ/Γ_f)} χ_γ δ_{γ∈Γ_f}` | flavoured BPS charts ("Flavour in `KAlgebra`"); `BPSKAlgebra.trace` aggregates the central-direction R-content (`snf_kernel` extracts `Γ_f = ker B`) |
| Rem. `rem:flavour-comments` | line defects assumed `G_f`-equivariant; the lift is canonical up to a 1d irrep; one may need to extend the lines or replace `G_f` by a finite cover; disconnected `G_f` (a `Z_5`-equivariant pentagon trace) left open | Plan 32 (`restructuring_plans/32_flavour_forgetful_lift/decisions.md`): the finite cover is A5, SO(3) → SU(2), which turned the virtual half-integer characters into honest irreps; disconnected `G_f` has no repo counterpart |
| `sec:uqsl2`, Definition 2.13 (unlabelled), `eq:uqsl2trace`, `eq:uqsl2rec`, `eq:uqsl2con`, Remark 2.14 | `U_𝖖(sl_2)`'s central quotient as an `SU(2)`-flavoured `K_𝖖`-algebra (= SQED₂): relations, `E_{a,b} = 𝖖^{−ab}E^aK^b`, `F_{a,b} = 𝖖^{ab}F^aK^b`, Lusztig's braid `ρ` of infinite order, `G(x,μ) = Σ_n Tr K^n x^n = (𝖖²;𝖖²)²_∞ E_𝖖(μx)E_𝖖(μ⁻¹x)E_𝖖(μx⁻¹)E_𝖖(μ⁻¹x⁻¹)`, the four-term recursion; ADE / global-form variants expected | algebra side: `implementations/uq_sl2_pbw.py` (relations, PBW canonical basis, `rho` = the braid; its `trace` deliberately raises — the Cartan trace needs the flavour layer).  Trace side: `G(x,μ)` and the recursion are `trace_bootstrap.md` §(B)–(C), triple-validated there against `Sqed2SU2OverPure.trace` (now `legacy/sqed2_su2_over_pure.py`, retired 2026-09-19 in #1510 — the measurement stands); the live SQED₂ is `GNAbeKAlgebra(u_n(1), (1,), nf=2)` (`R(U(2))`, `base_change(un_to_sun_hom(2))` for `R(SU(2))`), whose `trace` IS `eq:measure` at `(U(1), 2 fund.)`, i.e. the draft's own derivation of `eq:uqsl2trace` — **reproduced exactly through `𝖖⁹` at `n = 0, ±1, 2`, 2026-09-21** (the positive-check list below).  Remark 2.14: `uq_sl3_pbw.py`, `uq_sln_pbw.py`, `uq_sln_root_pbw.py`, Plan 27.  The footnote (the bar here fixes `K` and is antimultiplicative; the basis is not Lusztig–Kashiwara's) is ONBOARDING.md "False friends" verbatim |
| (rank 2, no draft section yet) | `U_𝖖(sl_3)` as an SU(3)-flavoured `K_𝖖`-algebra | `implementations/uq_su3_kalgebra.py::UqSU3KAlgebra` — **labels since 2026-09-26 (Plan 27 ruling A20): `(σ, λ)`**, `σ` a PBW monomial divisible by neither leading monomial of `χ₁`, `χ₁₁` and `λ` an SU(3) irrep, `L(σ,λ) = χ_λ·L(σ)`, built by Kazhdan–Lusztig + Webster's correction with no Cartan-window cap and traced to any 𝖖-order under a certified truncation; `≅` the `GNAbeKAlgebra` oracle as the composite `uq_su3_abe_iso` through the private `(m, e)`-keyed construction `_UqSU3MERoute`.  The rest of this cell describes that construction, which was the public class until A20 (2026-09-21): the U(1)−U(2)+3 algebra in Chevalley letters with Cartan generators `K₁ = W(1;0,0)`, `det₂ = W(0;1,1)` and the U(2) fundamental Wilson lines as derived Gelfand–Tsetlin Hamiltonians (Plan 27 A10); free PBW engine on `E₁ < E₁₂' < E₂ \| K \| F₂ < F₁₂'' < F₁`, canonical basis on the Borel halves = Kazhdan–Lusztig basis of the 𝖖-normalised PBW basis (`𝖖^{−b(a+c)}E₁^aE₁₂'^bE₂^c ↦ L_{(a+b;b+c,0),(b,−b,0)} + 𝖖Z[𝖖]·higher-b`), Cartan sector = U(1)×U(2) characters, trace = closed Schur-index integrand; certified `≅` the `GNAbeKAlgebra` oracle by `implementations/uq_su3_abe_iso.py`; stage 2 (2026-09-22) builds the mixed sectors constructively (Wilson-dressing solves, Kazhdan–Lusztig extraction from the product whose lowest term is the target, ρ-mirrors); stage 3 (2026-09-22) builds EVERY label — Kazhdan–Lusztig extraction from a charge-peeling product (raw layer) + Webster's Gram–Schmidt under the Schur pairing with `⟨L,L⟩ = 1 + O(𝖖)` (canonical layer), the pairing from a trace on PBW words derived from the Cartan monomials and ρ²-twisted cyclicity alone (`_word_trace`), so `inner_product` works on every label; certified vs the oracle: 14 labels of the general route exact against the oracle at root length ≤ 5 (sectors (0;2,−1), (0;1,−2), (0;2,−2), (0;3,0), (0;2,1), (0;0,−2), (0;−1,−1); every witness product expanded in the oracle), `Tr(T₂)` through 𝖖⁴ and the pairings `I(E₂,E₂)`, `I(T₂,T₂)`, `I(T₂,K₂⁻¹)` equal to the oracle's; labels of root length ≥ 6 build but the lazy raw cascade is slow there (open); third pass (2026-09-22): class lines by the Wilson witness `L_{m,e'}·w₂` (the exact 2×2 solve holds one class off a closed-form label only — the element's own bubbling enters the product, measured on the oracle), F-side sectors as ρ-images, the root-length law `r = r_base + 2k` with closed-form `r_base` and the base interval (= the Kazhdan–Lusztig range of the halves) as the lookup filter and the guard on every general-route result (an element of another root length is another label wearing the name), the peel consulting the lookup for every top word under the product-content rules; open: the Levi roots with U(1) charge (`E₁·L_{m,e} = Σ_S 𝖖^{c_S}L_{e+w_S}` over subsets of the bifundamental weights, so `(1;1,1)`, `(1;2,1)` are not one-product extractions) and the `(0;k,k)` roots, `k ≥ 2` (Plan 27 notes, "third pass"); the positive half is the quantum cluster monomial basis |
| `sec:a1d3`, Definition 2.15 (unlabelled), `eq:a1d3trace`, Remark 2.16 | `K_𝖖([A_1,D_3])`: six generators `T_i, D_i`, the `Z_3` `ρ`, the three elementary traces over the Weyl–Kac denominator `D(μ,𝖖)` | `a1d3_kalg.py` (**checked 2026-09-21: `Tr 1 = 1 + χ_2𝖖² + (χ_0+χ_2+χ_4)𝖖⁴ + …`, `Tr T_0 = −𝖖 + 0·𝖖³ + …`, `Tr D_0 = −χ_1𝖖 − χ_3𝖖³ + …` agree with `eq:a1d3trace` at the orders expanded by hand**), `finite_a1d3_kalg.py`, `a1d3_cone_data.py`, `a1d3_object.py`; the family `[A_1,D_{2n+1}]`: `a1dn_kalg.py`, `a1dn_atlas.py` |
| `sec:coulomb`: labels `(m,e) ∈ (Λ_m × Λ_e)/W`, `m` dominant, `e` dominant for the Levi `𝔤_m`; the footnote ('t Hooft–Wilson lines) | the canonical labels of `K_𝖖[𝔤,M]` | the `(m,e)` label frame of `PureGAbeKAlgebra` / `GNAbeKAlgebra` (CLAUDE.md, `AbeKAlgebra.fold`); `wrq_torus.rho_label` acts on Weyl orbits of pairs |
| `sec:coulomb`: global forms — `Λ_G ⊂ Λ^w_m × Λ^w_e` Lagrangian; the minimal `K_𝖖[𝔤,M]` inside the maximal `K_𝖖[G,M]` | the 4d gauge group as a maximal mutually local set of `(m,e)` | `global_form.LineLattice` (Plan 24 D10/D19): `verify_dirac_integral`, `verify_maximal`, `langlands_dual`; the `lines=` argument of both classes |
| `sec:coulomb`: flavour enhancement (`U(1)^n → U(n)` for `n` copies; `USp(2n)` / `SO(2n)` for (pseudo)real irreps; `R ⊕ R^∨`) | which `G_f` a matter content carries | `UNZPlusRing` / `SUNZPlusRing` (Plan 22 D5: `SU(N_f)` per node, reached by `base_change(un_to_sun_hom)`), `flavour_enhancement.py`; the `SO(2N_f)` enhancement of `SU(2)+N_f` is the recorded follow-up (CLAUDE.md `AbeKAlgebra` entry) |
| `eq:rho-witten`, `κ(m)`, `κ_f(m)` | `ρ : (m,e) ↦ (−m, κ(m) − e)`, `κ` the difference of the adjoint and matter sums over weights positive on `m`; `κ_f` the flavour shift | `wrq_torus.rho_label` (the explicit closed form; CLAUDE.md `AbeKAlgebra` entry), `rho_level_star` (`w_i ↦ w_i^⋆ ⊗ det_i^{−D_i}` is `κ_f`); at `N = Adj` the antipode |
| "The Abelianized presentation": `f_m(v) = v^e g_m(v^α)`, denominators `(1−𝖖^k v^α)`, bar with `v` fixed | the rational quantum torus | `WRQTorus` / `MatterWRQTorus` residual families on `weyl_torus_ring.TorusRational`; `bar`; `well_formed_w1` |
| `eq:fgprod` | the product on residuals `f_m(v)` | `WRQTorus.__mul__`, `MatterWRQTorus.__mul__` |
| `eq:ccprod`, `eq:ccclosed` | the cocycle: one vector factor per positive root, one hyper factor per weight, closed Clebsch–Gordan range (`p = min(|A|,|B|)`, `r = ||A|−|B||`) | `wrq_torus.CC`, `CC_root_factor` (verbatim), `cocycle_range`; `matter_wrq_torus.CC_N`, `_matter_factor`, `matter_factor_exponents` |
| `eq:cocydef`, `f = Σ_m f_m(𝖖^m v)U_m`; the remark that the compact notation may carry fractional powers of `𝖖` for some `Λ_G`, cancelling in the end | `U_a U_b = CC_{a,b}(𝖖^{a+b}v) U_{a+b}` | `cocycle_R` (`R = T_{a+b}(CC)`); the repo's f-presentation uses the same `f_m(𝖖^m v)` convention; the fractional powers are never materialised — D31 (only the integral coboundary enters `cocycle_R`) |
| `eq:rhotorus` | `(ρf)_{−m} = v^{κ(m)}μ^{κ_f(m)} f_m(1/v, 1/μ)` | `WRQTorus.rho` / `MatterWRQTorus.rho`; the monomial twist `G̃_m = v^{Σ_{α>0}⟨α,m⟩α}` is derived in `sector_measure_closed_form` (step 1), the level star in `matter_sector_weight` |
| `eq:measure` | the trace as a contour integral with the Schur / hyper measure, prefactor `(𝖖²;𝖖²)_∞^{2 rk 𝔤}/|W|` | `trace_residual` (pure), `MatterWRQTorus._mu_trace_rows` (the Nahm factor `1/(−𝖖x;𝖖²)_∞`) |
| `eq:Iexplicit` | `I_{f,g}` with `|⟨m,α⟩|`-, `|⟨m,w⟩|`-shifted measure and no rational factor | `inner_by_sector` on both tiers; `sector_measure_closed_form` / `matter_sector_measure_closed_form` — `T_{+m}μ·B_m` and `T_{+m}μ_N·∏Ξ_c` ARE this measure (pinned per root pair and per weight, `tests/test_sector_measure_closed_form.py` leg 6; Plan 24 D53) |
| remark after `eq:Iexplicit` | `I = Tr ρ(f)g` holds at integrand level up to contour shifts `v → 𝖖^{±m}v`; pole compatibility conjectural | the "contour move `ι = v̄∘T_{2m}`" item of the axiom-5 bullet above: measured, not derived; the measure crosses no pole (`M_m` entire), so the conjecture is about the residuals' poles.  The equality with `Tr ρ(f)g` (multiply-then-trace) is checked in Plan 42 row `eq:Iexplicit` at SU(2), SO(3)+adjoint, SU(3) and SU(3)+1 (2026-09-23), beside `tests/test_wrq_sector_pairing.py` |
| `eq:cocha`, `d_a = f_a(𝖖^a v)·cocha_a(v)` | the trivialisation with the `(−𝖖)^{½Σ⟨α,m⟩}` prefactor — now written over the roots POSITIVE ON `m` (the Weyl-covariant form; the earlier all-`α>0` version is commented out in the source) | `dressing_psi` is the FLOORED `ψ` (the fractional `(−𝖖)` power stripped, user 2026-08-24), computed at the dominant representative and Weyl-transported — which is exactly the roots-positive-on-`m` form; the honest phase is restored through the integral coboundary in `cocycle_R` (D31); `cocycle_R_via_psi`, `verify_psi_trivializes_cocycle`; the `d`-form is the (★) frame of `star_bubbling`.  The draft's formulas are consistent among themselves: the printed `cocha` (matter factors included) trivializes the printed `eq:ccclosed` / `eq:ccprod`, `cocy_{a,b}(𝖖^{a+b}v)·cocha_{a+b}(v) = (−𝖖)^{s(a)+s(b)−s(a+b)}·cocha_a(v)·cocha_b(𝖖^{2a}v)`, the net exponent an integer (checked 2026-09-23 on 18 theories, Plan 42 row `eq:cocha`); the older all-positive-roots form fails it |
| the leading Weyl orbit `f_{w·m} = w·χ^{𝔤_m}_e` and the "bubbling correction" | the two contributions to `L_{m,e}` | the W2 seed (the Levi character) and the solved residuals of `star_bubbling` / `matter_multislot` |
| Conj. `conj:abeKalgebra` A1 (`ab:bar`) | bubbling Weyl-covariant and bar-invariant | W1 of `certify_canonical` (`well_formed_w1`); Weyl covariance is built into the residual family |
| A2 (`ab:ofq`) | bubbling `O(𝖖)` | W2 of `certify_canonical`; the `O(𝖖)` guard of the licensed solve |
| A3 (`ab:res`, `eq:star`) | `d_a` has simple poles only; `Res_{v^α=𝖖^{2l}}[d_a + d_b] = 0`, `b = s_α(a) − lα^∨` | (★): `star_bubbling.criterion`, `wall_partner`, `residue_cancellation_recognition.md`; `matter_star_bubbling` with matter; the production route of `PureGAbeKAlgebra` / `GNAbeKAlgebra` (D40).  ⚠ `criterion` tests that each pair sum is regular on the wall, which is the residue cancellation when the poles are simple; the clause "simple poles only" is imposed by no solver and checked by no verifier — it is MEASURED (2026-09-23: every `d_a` of every window line has simple poles, pure and with matter; Plan 42 row `conj:abeKalgebra/existence-uniqueness`).  With matter the code imposes the rule on the dressed image `F = Z·f` through its own trivialization `Z`; that the stored lines also satisfy the draft's A3 with the printed matter factors of `eq:cocha` is checked in the same row, and the weights of `N^∨` in place of `N` fail it on complex matter |
| note after Conj. 1 | the support on Weyl orbits subdominant to `m` "appears to follow from the above axioms"; the `d`'s may carry a common fractional power of `−𝖖` | `star_bubbling.tropical_support` (the support is subsumed by (★) — measured, "The `L_{m,e}` construction" above); D31 |
| "The `K_𝖖` algebra axioms" (subsection after Conj. 1) | Klyuev's residue rule; equivalently a well-defined action of the rational torus on the characters, `u^m χ_e(v) = χ_e(𝖖^{2m}v)`; the conjecture may reduce to "bar-invariant with `O(𝖖)` bubbling"; `O(𝖖)` bubbling + `eq:Iexplicit` ⇒ `δ_{a,b} + O(𝖖)`; the category `KP(G,N)` | (★) ⟺ the line preserves `Λ = R(G)` of this form (`residue_cancellation_recognition.md` §2/§5 — the same statement); the one-route ruling of 2026-08-25 is the simplification taken; the `δ + O(𝖖)` implication is how the sector pairing reads (`inner_by_sector`, "inner_product — the Schur pairing as a sum over sectors"); `KP(G,N)` has no repo counterpart |
| `eq:su2so3lat` | the three rank-1 global forms (`sl_2`: both even; `SU(2)`: `m` even; `SO(3)`: `e` even; the third: `e − m` even), Dirac pairing `½(me′ − m′e)` | `global_form.LineLattice` (Plan 24 D10/D19): `simply_connected_lines(su_2())`, `adjoint_lines(su_2())`, `so_n(3)`; the third form via `lines=`.  Coordinates: the repo's `su_2()` writes `m` in COROOT units, so the draft's `L_{2,0}` is the repo label `((1,), (0,))`; `so_n(3)` writes `m` in coweight units (root `(1,)`), so the draft's `L_{1,0}` is `((1,), (0,))` there |
| `eq:su2thooft`, `eq:su2dform`, `eq:su2res`, `eq:su2dyonic` | `L_{0,2}`, `L_{2,0} = L_{1,0}²` with `f_0 = (𝖖+𝖖⁻¹)v²/((1−𝖖²v²)(1−𝖖⁻²v²))`, its `d`-form and the three residue cancellations, `L_{2,1}` | `PureGAbeKAlgebra(su_2())` (**checked 2026-09-21, residual for residual: `((1,),(0,))` is `eq:su2thooft`, `((1,),(1,))` is `eq:su2dyonic` once the draft's `f_m(𝖖^m v)` argument shift is applied to the stored residuals, `((0,),(2,))` is `L_{0,2}`**), `PureSU2KAlgebra`; `tests/test_pure_g_abe_kalgebra.py`; the residue cancellations are `star_bubbling.criterion` / `wall_partner`; "a Laurent polynomial cannot be bar-invariant and `O(𝖖)`" is the W1/W2 rigidity |
| `eq:adjsquare`, `eq:adjbubble`; `κ ≡ 0`, `κ_f(m) = −m` | `N = 2*` `SU(2)` / `SO(3)`: `L_{1,0}·L_{1,0} = L_{2,0} + μ`, `f_0(L_{2,0})` | `GNAbeKAlgebra(so_n(3), (1,), nf=1)` (the `SO(3)` form; **checked 2026-09-21: the product is `L_{(((2,),(0,)),(0,))} + L_{(((0,),(0,)),(1,))}`, i.e. `L_{2,0} + μ`, and the `m = 0` residual of `L_{2,0}` is `eq:adjbubble` term by term**) and `roster("su2-adjoint")` (`g_matter_roster.py`, the `SU(2)` form, same `f_0`); `rho_label` gives the antipode at `N = Adj` |
| Def. `def:rg` R1 (`ax:lattice`) | `Γ`-graded IR `K_𝖖`-algebra | `RGKAlgebra.grading()`, `auxiliary()` |
| R2 (`ax:map`) | injective bar-invariant `RG: L^UV_a ↦ RG_a`, an integral combination of `[n]_𝖖 L^IR_b`; footnote: the no-exotic (non-negativity) conjecture is not imposed | `RG(a)`, `verify_rg_multiplicative`, `verify_rg_bar_invariant`, `verify_rg_unital` — ⚠ INJECTIVITY has no verifier (repo_audit A93): immediate on `BPSKAlgebra` from the tropical readout, unchecked on a generic flow; non-negativity is not imposed here either |
| R3 (`ax:S`) | spectrum generator `S ∈ Z[𝖖,(1−𝖖^{2n})⁻¹] ⊗ K_𝖖[IR]_{Γ_+}`, `Γ_+` a finitely generated convex cone (footnote: "a bit artificial"), `RG_a S = S ρ⁻¹_IR(RG_{ρ_UV(a)}) = L^IR_{ℓ(a)} + O(𝖖)`, `ℓ` bijective (footnote: may be relaxed), `S = 1 + O(𝖖)` | `rg_generator` / `_s_rg_component` (`HabiroElement`: the `(1−𝖖^{2n})` denominators are the type), `apex` (= `ℓ`), `verify_rg_discovery`, `verify_rg_twist`, the co-solver `graded_rg_solver` |
| R4 (`ax:rgtrace`) | `(L^UV_a|L^UV_b)_UV = (RG_a S|RG_b S)_IR` | `verify_rg_inner_product`; axiom-5 inheritance `verify_rg_inherits_rho_star` |
| Remark 4.2 (reconstruction from `(IR, S)`) | `RG_a` determined by `S`; the UV algebra reconstructed from `(K_𝖖[IR], S)` | `RGKAlgebra`'s design principle (CLAUDE.md "Why this shape") |
| Remark 4.3 (composition) | `RG^{1→3} = RG^{2→3}∘RG^{1→2}`, `S^{1→3} = RG^{2→3}(S^{1→2})·S^{2→3}` | `then` / `factor_through` → `ExtractedRG` ("Factored / partial RG flows", Plan 21) |
| Remark 4.4 (the `ρ⁻¹` family) | `(RG, S) ↦ (ρ⁻¹_IR(RG_{ρ_UV(a)}), ρ⁻¹_IR(S))` yields new flows; `I_{a,b} = I_{ρ⁻¹(b),ρ⁻¹(a)}` is the safe rewriting | same algebra, different flow: `RGKAlgebraObject`; `bps_chart_object.chart_monodromy_iso` (the full rotation is `ρ²`).  The identity is K4's twist, not K5 |
| Def. `def:wallcrossing` | `S^{1 1̄} = S^{12}S^{2 1̄}`, `S^{2 2̄} = S^{2 1̄}ρ⁻¹_IR(S^{12})`, with the intertwining relations; partial factors need not be `O(𝖖)`-good | chart mutations as different flows of one algebra: `bps_chart_object.mutate_bpskalgebra`, `RGKAlgebraObject`, `BPSAtlas` (`verify_spec_equivalence`, `monodromy = ρ²`) |
| Remark 4.6 (central charge) | `Im e^{−iϑ}Z_γ > 0` gives a locally constant family of flows on the universal cover of `S¹`; `ρ^{±1}` is `ϑ ↦ ϑ ± π` | `bps_factor_spectrum`'s `phases=` (a linear central charge, `arg Z_γ`); `BPSAtlas` — the full rotation composes to `ρ²` (tested on the pentagon) |
| Remark 4.7 (matter removal) | RG to `K_𝖖[G,∅]⊗Q_𝖖[Γ]`, `S = ∏_w E_𝖖(μv^w)`; factors per summand of `N` | `GMatterOverPure` / `GMatterOverMatter`, `matter_removal_tower` (`S_RG` = the matter dilogarithm product `Ψ`) |
| "Seiberg–Witten RG flows": the pro-nilpotent group `𝓖`, `E_𝖖(x) = (−𝖖x;𝖖²)⁻¹_∞`, `eq:mult` (`E^{(s)}_𝖖`), Definition 4.8 (`𝓔`, unlabelled), Remark 4.9 | `𝓔` is generated by `E^{(s)}_𝖖(X_γ)^{±1}`; every `g ∈ 𝓔` has `g⁻¹(𝖖) = g(𝖖⁻¹)` but `𝓔` is smaller than that group (`E_{𝖖²}(x) ∉ 𝓔`) | `HabiroElement` coefficients on the quantum torus; the factors of `bps_factor_spectrum`; `pronilpotent_group_conjecture.md`; Goal 3.11 "the `S⁻¹(𝖖) = S(𝖖⁻¹)` identity" |
| Def. `def:sw`, `eq:quiver` | Seiberg–Witten flows (`K_𝖖[IR] = Q_𝖖(Γ)`, `S ∈ 𝓔`); BPS-quiver flows, `S = 1 − 𝖖ΣX_{γ_i} + O(𝖖²)` | `BPSKAlgebra`; spec finder `BPSQuiver.find_negating_sequence`; the dictionaries `dictionaries/enumerated`, `spec_acceptance.py` (the finite-depth check of `eq:quiver`, Goal 3.12) |
| Conj. `conj:S-unique`, `conj:pbw`, Remarks 4.11–4.12 | uniqueness of `S`; PBW factorisation in any total order on `Γ_+ × ½N`; uniqueness along the filtration and constructibility in every order; the physical order is `arg Z_γ` | `bps_factor_spectrum` (order = any total order on the pairs `(γ, s)`; kalgebra.md "S-finding"), `factor_order_search`, `palindromic_spectrum`; `BPSAtlas.verify_spec_equivalence` |
| Conj. `conj:cluster`, `conj:dt`, `conj:coha` | `S_cluster = S` (maximal green sequences), `S_DT = S`, `S_CoHA = S` | green-sequence replay in the enumerated dictionary; `BPSAtlas.verify_spec_equivalence`; `S_DT`, `S_CoHA` not encoded (`RESEARCH_GOALS.md`, subproject "Hall algebra vs COHA for the BPS factors") |
| the paragraph before Conj. 7 | given `S(Q)`, `RG_γ S = X_γ + O(𝖖)` has a unique bar-invariant solution supported on `γ + Γ_+` | the F-solver (`solve_F`, kalgebra.md "F-finding"); `fs_builder` builds `F_γ` and `S` together |
| Conj. `conj:upper` | `K_𝖖(Q,Γ)` = upper cluster algebra with canonical flow to `Q_𝖖(Γ)` | the F-solver (`bps_quiver_tools.solve_F`, `recursive_spectrum`), `BPSKAlgebra` chart graph, `BPSAtlas` |
| Conj. `conj:factored` and the closing question | factored flows `S_Q = RG^{Q',Γ}(S_Q^{Q'}) S_{Q'}`; which pro-nilpotent group hosts `S_Q^{Q'}` | `RGKAlgebra.factor_through` → `ExtractedRG`, `rg_flow.factor_through_subquiver`, `DirectionalSubquiverRG` (Plans 20/21); `pronilpotent_group_conjecture.md` |
| "Flavoured BPS quivers": `eq:fleq`, Definition 4.13 (`𝓔_{G_f}`, unlabelled), `eq:flavouredquiver` | `E^{(s;R)}_𝖖(x) = ∏_{w∈R}E^{(s)}_𝖖(μ^w x)`; `N` identical nodes ⇒ `SU(N)`-flavoured; `S = 1 − 𝖖Σ_i χ^{□_i}(μ)X_{γ_i} + O(𝖖²)`; PBW extended to `𝓔_{G_f}` | `flavoured_factor_spectrum`, `flavoured_f_solver`, `flavoured_spec`, `gbps_kalgebra.GBPSKAlgebra`, `flavoured_s_axioms.md` (the character-valued `Ω` is EMERGENT there, never imposed); the FLAVOURED dictionary family (`dictionaries/flavoured*`, `lookup_flavoured(orbit_size=N)`); `add_flavour` |
| `sec:skein`, Conj. `conj:junctions`, Remarks 5.1–5.2 | skein algebras as `K_𝖖`-algebras; canonical junctions; trace from `D²×C`; non-negativity of collapse coefficients fails; the conjecture proven in upcoming work (Baumann–Kamnitzer) | `skein_sphere/` (`SkeinKAlgebra`, `SkeinAtlas`, its `KQ_CONJECTURE.md`) |
| App. `app:physics` | physical origin of the axioms; the RG-interface pairing `(L^IR_a, RG_b S)_IR`; `eq:int`, an interface between `(G,N)` and `(T_G,N)` with one `|⟨m,α⟩|`-shifted vector factor per root | `residue_cancellation_recognition.md` §2/§5; `RESEARCH_GOALS.md`.  The RG-interface pairing is the RG-transport axiom's object (`interface_indices.md` §6, `RGInterface`, `uv_index_from_ir`).  `eq:int` has no certified counterpart: the nearest repo object is the one-dressed-slot pairing `(a|Π] = ⟨L_a·1, Π⟩` of `aux_vacuum_pairing.md` §1 against a torus monomial state — unverified against `eq:int` |
| App. `app:finite`, Table `tab:finite_type` | finite-type `K_𝖖`-algebras `[A_1,ADE]`: flavour symmetry and order of `ρ` (`[A_1,A_{2n}]`: `2n+3`; `[A_1,A_3]`: `SU(2)`, 3; `[A_1,A_{2n+3}]`: `U(1)`, `2n+6`; `D_4`: `SU(3)`, 4; `[A_1,D_{2n+1}]`: `SU(2)`, `2n+1`; `[A_1,D_{2n+2}]`: `SU(2)×U(1)`, `2n+2`; `E_6`: 14; `E_7`: `U(1)`, 10; `E_8`: 16); `[A_1,E_6] ≅ [A_2,A_3]`, `[A_1,E_8] ≅ [A_2,A_4]` | `finite_type_kalgebras.md` (§8 records the measured orders 5, 4, 14, 16 and 10 for the pentagon, `D_4`, `E_6`, `E_8`, `E_7` — all agree with the table; its "`h+2`, halved where `−w₀ = id`" reading is the `[A_2,A_k]` corner's, not general: `[A_1,D_{2n+1}]` has order `2n+1` with `−w₀ ≠ id`, e.g. `a1d3_kalg.py`'s `Z_3` at `D_3 = A_3`; `u1a1aodd_kalg.py`'s `U1A1AoddKAlg(2)` has the `ρ`-orbit of `[A_1,A_5]` closing at period 8 modulo the `U(1)` charge, as the stand-alone `u1_octagon_kalg.py` (in `legacy/` since 2026-09-23) recorded), `finite_kalgebras.objects`, `finite_e6_kalg.py` / `finite_e7_kalg.py` / `finite_e8_kalg.py`, `finite_a1d4_kalg.py`, `a1dn_kalg.py`, `hexagon_kalg.py` |
| App. `app:a1a2k`, Figure `fig:heptagon-orbits` | odd polygons `K_𝖖([A_1,A_{2k}])`: chords `L_{a;i}`, the `c_{ab}` chord rule, quantum Ptolemy with `(α,β) ∈ {(1,0),(0,−1)}`, the trace reduction algorithm, `T_a = (−1)^{m+1}𝖖^{−m}(χ_m − χ_{m+1})(𝖖²)` with `m(a)`, the minors miracle | `a1a2k_cone_data.py`, `a1a2k_kalg.py`, `a1a2k_kq_proof.md` (K1–K3 proofs and the trace program), `a1a2k_axioms_at_all_k.md`, `a1a2k_all_orders.md`, `a1a2k_leading_multiplicativity.md`, `polygon_trace_decompose.py`; the miracle rows above |

Two places where the draft and the repo deliberately differ in presentation:
the draft's `cocha_m` carries the honest fractional `(−𝖖)` prefactor while the
repo ships the floored `ψ` and restores the phase through the cocycle (D31 —
same algebra, integral surface); and the draft's pairing measure is written
with absolute-value shifts while the trace machinery here carries the signed
shifts plus the rational `B_m` / windows `Ξ` — proved equal above.

**Positive checks run against the 2026-09-21 version** (each a few seconds
from the repo root; rerun them when the draft changes):

* `tab:pentagon-stabilization`: `implementations/pentagon_trace.t_coeffs(n)`
  for `n = 3..6`, multiplied by `𝖖^{n²−1}` — all four rows exact.
* `eq:a1d3trace`: `A1D3KAlg().trace` on the identity, `T_0` `(0,1,0,0)` and
  `D_0` `(0,0,1,0)` at `K = 8` against the hand expansion of the three
  Weyl–Kac quotients — agree through the orders expanded (`𝖖⁴`, `𝖖³`, `𝖖³`).
* `eq:su2thooft` / `eq:su2dyonic`: `PureGAbeKAlgebra(su_2()).chart` at
  `((1,),(0,))`, `((1,),(1,))`, `((0,),(2,))` — identical residuals.
* `eq:adjsquare` / `eq:adjbubble`: `GNAbeKAlgebra(so_n(3), (1,), nf=1)` —
  `multiply` of the minuscule 't Hooft line with itself, and the `m = 0`
  residual of `L_{2,0}` (also on `roster("su2-adjoint")`) — exact.
* `eq:uqsl2trace`: the live SQED₂ `GNAbeKAlgebra(u_n(1), (1,), nf=2)` Wilson
  traces `(((0,),(n,)),(0,0))`, after `base_change(un_to_sun_hom(2))`, against
  the coefficient of `x^n` in `G(x,μ)` expanded in `𝖖` — exact through `𝖖⁹` at
  `n = 0, ±1, 2`.  Note `Tr K⁰ = 1 + 𝖖²(χ₂ − 1) + …`: the `−1` is the U(1)
  vector multiplet's `(𝖖²;𝖖²)²_∞` (`trace_bootstrap.md` once printed
  `1 + 𝖖²χ₂ + …`; corrected 2026-09-21).

Not encoded anywhere in the repo, and said so above: Remark 2.3 (Webster),
the disconnected-`G_f` case of `rem:flavour-comments`, the category
`KP(G,N)`, `S_DT` and `S_CoHA`, and `eq:int`.
