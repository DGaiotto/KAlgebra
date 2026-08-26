# GNKAlgebra — gauge theory at **any** 4d gauge group, with **any** matter

**Step 7**, the general-gauge-group layer of this repository (`src/gn/`). It
answers the question Steps 1–6 leave open: *which* gauge theories can the
abelianized tier actually present?

Steps 1–6 build the contract, the cone tier, the RG engine, the BPS spine, the
abelianized tier and the skein realisations — but the abelianized keystone leans
on a luxury that is special to type A. In `U(N)` and `SU(N)` every fundamental
coweight is **minuscule**, so dressed minuscule monopoles are closed-form
bar-invariant generators and every canonical is a polynomial in them. Outside
type A that fails, and it fails hard: `Spin(5)` has **no minuscule cocharacter
at all**, so every monopole bubbles and product-and-peel has nothing to start
from. Step 7 is what removes that restriction.

## This layer relies on Steps 1–6 (by design)

Steps 1–3 are deliberately spine-free; Steps 4–7 are the opposite. Step 7
imports the core contract (Step 1), the cone tier (Step 2), the RG engine
(Step 3) and the abelianized tier (Step 5) by bare name, with nothing
duplicated — importing a Step-7 module without the earlier layers raises
`ModuleNotFoundError`, so the dependency is real rather than a copy. It also
**bumps** seven earlier-tier modules; see "What this step updates".

The BPS spine (Step 4) is reached by **exactly one** module of this tier, and
measured rather than assumed: `pure_so3`, the independent oracle described under
"What it ships", is the only file here that imports anything from `src/bps/`. The
working tier — `PureGAbeKAlgebra`, `star_bubbling`, `GNAbeKAlgebra` and the rest —
does not, so the general-`G` machinery is not built on a BPS realisation; it is
*checked* against one. That is also why this suite runs after the Step-3 suites
that assert no spine module is loaded.

## What it ships

### H. Pure gauge at an arbitrary root datum

`pure_g_abe_kalgebra.PureGAbeKAlgebra` — one class over a `RootDatum`, with
labels `(m, e)` (`m` cochar-dominant, `e` Levi-dominant, the lower-Kapustin
convention).

**One route builds every label.** The axioms alone — the W2 seed, bar
(`W1`), `O(𝖖)`, and (★) — pin `L_{m,e}` at every datum, every matter content and
every label, and `star_bubbling.solve_canonical` is the single production route.
`route(label)` reports what fired.

*This retracts what an earlier revision of this page published:* a six-row
ladder `wilson → minuscule → cone → peel → monoid → star`, described as
"the constructive routes come first and each is separately guarded; the solve is
the last resort", together with a table naming the four charges at which the
licence "actually fires". Both are void. The licence fires everywhere, and the
table stated a property of a ladder that no longer exists. Correcting a
published claim is what a version increment is for.

**Wilson is the degenerate case, not a special case.** At `m = 0` the tropical
support is the single cell `{0}`, so the system has **no unknowns**: the W2 seed
*is* the element, `L_{0,e} = χ_e(v)·U_0`. `W1` holds because `χ_e` is `𝖖`-free,
and (★) holds because `R(G)` is a ring — both add zero constraints there. So the
Wilson lines follow from the axioms too, rather than from a route that knows
about them.

Two qualifications, because they are where the content sits: `χ_e` enters as the
**seed**, i.e. as the W2 normalization — it is not solved for; and electric
admissibility of `e` at a non-simply-connected form is enforced by the gauge
group data inside `chart`, **not** by (★), which tests against the datum's own
`R(G)`.

**The axioms are executable, in one place.**
`star_bubbling.verify_axioms(datum, m, e, x)` runs all five against a *finished*
element and returns `{condition: (ok, detail)}`. It exists apart from the
solver's internal guard because the guard runs on what the solver just built,
whereas this runs on anything — so "this element satisfies the axioms" becomes a
test rather than a restatement of how it was made. What it does **not** certify
is uniqueness: that is a property of the system, not of an element, and it stays
the solver's own first guard.

Each condition catches something the others do not, and the gate pins that with
**negative controls**: a bubbling-stripped `SU(2)` `L_{(2,),0}` passes W1, W2,
support and `O(𝖖)` and is rejected by **(★) alone**; a good element under the
wrong `e` is rejected by W2 alone; `L_1·L_1` asked as `L_1` is rejected by
support, W2 and `O(𝖖)` together.

**The support bound is not a sixth axiom** — it is subsumed by (★). Graft onto a
genuine canonical a bubbling residual at a cell *outside* `conv(W·m)`: bar acts
cellwise so such a residual is palindromic, and a bubbling residual has
`val_𝖖 ≥ 1`, so the perturbed element satisfies W1, W2 and `O(𝖖)` **by
construction** and (★) is the only axiom that can see it. It sees it every time —
30/30 grafts rejected, 0 accepted, at SU(2), SU(3) and Spin(5), each with the
true canonical passing as the positive control. The mechanism: (★) is a residue
cancellation between a cell and its affine-Weyl reflection partners, so it is the
one condition that couples cells *across* magnetic charge. This is evidence, not
proof.

**Two declared optimizations**, and they assume nothing about `G` or `N`:
`optimizations=` switches them per name, and `optimizations=()` reproduces the
axioms-only build exactly, so *"the axioms alone build every label"* stays a
**runnable** claim rather than a historical one.

| name | applies to | licensed by | measured |
|---|---|---|---|
| `theta_twist` | any label `T`-related to one already in the chart memo | the `e ↦ e + k·m̄` symmetry `T^k : f_p ↦ v^{k·p̄}f_p` — **measured** 30/30 and *not proved*, so the route stays (★)-guarded | identical element 14/14; 253× at SU(3) `m=(1,1)`, **18 370×** at SU(3) `m=(2,2)` (54.2 s → 2.9 ms), **29 540×** at `G₂` `m=(2,3)` |
| `monoid` | undressed `L_{m,0}` at a splittable dominant `m`; **pure gauge only** | a theorem: `L_{m,0} = (−1)^{⟨ρ,m⟩}𝖖^{⟨ρ,m⟩}Λ_m` with `Λ_m : P_μ ↦ 𝖖^{2⟨μ,m⟩}P_μ` on a common eigenbasis, so eigenvalues multiply and the normalizations cancel | identical 4/4; **345×** at SU(3) `m=(2,2)` |

The θ-twist is general in `e` — `T^k` carries `L_{m,e}` to `L_{m,e+k·m̄}` at *any*
`e`, in both directions — so the electric labels at fixed `m` fall into
`m̄`-lines and one known point gives the whole line. That is what ties it to the
memo: it multiplies the value of each existing entry rather than adding entries.

`monoid` is **not** offered to the matter tier: a free monoid cannot carry
Littlewood–Richardson multiplicities, so `L^N_{m,0}·L^N_{m',0} = L^N_{m+m',0}` at
`(G, Adj)` would be badly wrong. The name is still accepted there and governs the
inner pure-`G` algebra, which genuinely is the `N = 0` theory.

**The retired constructive routes are not erased.** `constructive_routes=True`
restores the whole zoo ahead of the solve, exactly as it was, and it is kept for
the reason the peel `S`-recursion is kept: it is an *independent* construction of
the same element, which is what makes agreement evidence rather than tautology
(33/33 over the label sweep). One route is excluded even there: `closed_form`
returns a **wrong element without raising** at SU(3) `m=(2,2)`, failing W1 *and*
(★) — its `one_step_cells` means one *root* below `m`, not one simple coroot, so
`m − θ^∨` is applied at an invalid depth. A route whose correctness lives in the
guard rather than in itself is the failure mode the constructive-build rule
exists to prevent.

**The `L_{m,e}` are memoized and persistable.** `save_cache` / `load_cache` on the
tier, the same names and shape as the RG tier's. The file is exact — an integer
numerator over an explicit denominator multiset — so a round trip is an identity.
Two guards, because a cache file is untrusted input and a *fast wrong answer* is
this tier's dangerous failure mode: the header fingerprints the presentation,
with the phase convention **probed rather than described** (a datum's phase is a
callable and cannot be compared any other way), and every admitted chart is
re-run through the axiom battery. That is affordable precisely because verifying
is not solving — measured at ~5% of a rebuild (SU(3) `m=(2,2)`: 4.67 s to build,
0.27 s to load with full verification).

### The (★) axiom — why a solve is licensed here

On the abelianized tier a canonical is normally assembled **constructively**, as
a polynomial in already-built canonicals, and where no closed form reaches, the
tier **honest-fails** rather than fabricating. The reason is sharp: a
linear-system solve can land **outside** the span of the canonical basis, and
both the orthonormality pairing and the M-test are pairings defined *within* that
span — so both pass on off-span garbage. A solve with no post-hoc guard is
therefore how you get silently wrong mathematics.

**(★) is that missing guard**, and it is a condition on this tier's own 4d
objects:

    (★)  ⟺  the element's `𝖖`-difference operator preserves `Λ = R(G)`

— the representation ring of the gauge group; equivalently, the element acts on
symmetric Laurent polynomials in the abelianized gauge fugacities. Concretely it
is the affine-Weyl residue cancellation between a cell and its reflection
partners, which is what `star_bubbling.criterion` tests.

That is a **membership condition on the presentation**, not a pairing inside the
span — which is precisely why it catches what orthonormality and the M-test
cannot. The sharpest statement of that is the derivation in
[`conjectures-step5-abe.md`](conjectures-step5-abe.md): orthonormality at `𝖖⁰`
**cannot see the bubbling at all**, because every non-leading cell is `O(𝖖)` by
axiom and the pairing weight cannot lower it. So a candidate missing its bubbling
passes the `𝖖⁰` self-norm and is rejected by (★) — measured 23/23. The two facts
are the same fact.

`R(G)` is the representation ring of **this global form**, not of the cover: at
`G = G̃/H` only the centre-neutral `e` are admitted, so `Λ_H = R(G̃)^H = R(G̃/H)`
is a *proper* submodule of the cover's (4 of 10 dominant weights at PSU(3), all
of them at a simply connected form). Asking (★) against the cover's `R(G̃)` at a
non-simply-connected form is a real bug, not a convention.

*Provenance.* The condition was **recognised** through a boundary argument: the
lines on a half-BPS boundary are generated by the images of the Wilson lines
`L_{0,e'}`, and each such image is exactly the character `χ_e`, so those images
span `R(G)`. What survives into the condition is the **ring** — the boundary
drops out. Nothing in this release needs a boundary condition to state (★) or to
test it, and none is constructed anywhere in it; the machinery that argument was
first seen through is Step 8.

### I. `(G, N)` matter at any datum and any matter representation

`gn_abe_kalgebra.GNAbeKAlgebra` — gauge theory with `T^*N` matter. **Matter is a
representation, not a count**: an `int` is shorthand for that many copies of the
defining representation, otherwise highest weights are given, so `Sp(4)+4`,
`Spin(5)+5` and `Spin(5)+4ˢ` (spinor) are distinguishable theories rather than
three things all declaring `1`. Also here: the constraint calculus
(`matter_star_bubbling`, `matter_multislot`), the **iterable** matter-removal
flows (`g_matter_over_matter.matter_removal_tower()` walks `SU(2)+3 → +2 → +1 →
pure`, each rung a certified flow over the previous rung's IR), named theory
**presets** (parameter tuples, not classes — a special class name is earned by an
algorithm optimized for a particular `G`/`N`), and the type-A **seam**
(`g_matter_un_nf_seam`), which is a **surjection, not an isomorphism**: even at
`N_f = 1`, where structure constants and `ρ` agree, the *trace* separates the two,
because the matter fugacity grades matter zero modes.

### J. The 4d gauge group data, and Langlands duality

`global_form.LineLattice` — the 4d global form as **a maximal set of mutually
compatible Kapustin `(m, e)` labels**, i.e. a lattice and its dual, with duality
verified *both* ways. This is the 4d notion; "which cocharacter lattice" is the
3d reading. So `SU(2) = (Q^∨, P)` and `SO(3) = (P^∨, Q)`, and in SU(2)'s
coordinates that index-2 inclusion `Q ⊂ P` reads "magnetic doubled, electric
halved" — a description of two lattices, **not** a rescaling.

`langlands_iso` supplies the Langlands family `(G, Adj) ↔ (G^∨, Adj)` as
certified `KAlgebraIso` objects, including the cross-datum `B₃ ↔ C₃`.

**A global form need not be a product.** The traditional forms are the *product*
case, (coweights of `G`) × (weights of `G`). In general the label space is the
Weyl quotient of a lattice `Λ` between `Q^∨ × Q` and `P^∨ × P` on which the Dirac
pairing is integral, so `Λ` is the preimage of a **finite** subgroup
`L = Λ/(Q^∨ × Q)` of centre-class *pairs*, and the Dirac pairing descends to the
linking pairing there:

    admissible  ⟺  L isotropic,        maximal  ⟺  L Lagrangian,

and **maximality is not required** — a non-maximal isotropic `L` is a consistent,
merely incomplete, set of lines. `W` acts trivially on both quotients, so every
admissible `Λ` is automatically `W`-stable and the dominant-representative frame
is already a section of `W\Λ`: the Weyl quotient costs nothing.

`LineLattice` takes either `H=` (a centre subgroup, the product forms) or
`classes=` (generators of an arbitrary class-pair subgroup). The general case
covers the **correlated** lattices — a discrete theta angle — and at `su(2)` it
gives *three* forms, not two: `SU(2)`, `SO(3)`, and the one requiring odd
electric weight when the magnetic weight is odd, whose only extra line is the
dyon `(ω^∨, ω)`. Class labelling is derived from Smith normal form rather than
from a searched generator, so non-cyclic centres (`Spin(4k)`'s `Z₂ × Z₂`) work.

⚠ Consequently `admits` is **no longer** `mag_admits and elec_admits` — that
conjunction is right only for a product form. The two side predicates are now the
*fibres* of `admits` over `e = 0` and `m = 0`, unchanged on every product form.

`line_lattice_torus` carries the lattice's own quantum tori.
`conventional_quantum_torus(lines)` reads the Dirac matrix on a `Z`-basis of `Λ`
and hands it to the `QuantumTorusKAlg` of Step 1 — integral exactly when the
lattice is admissible, so an inadmissible lattice is refused by *arithmetic*
rather than by a separate check. Two rational siblings differ **only** by the
cocycle: `DRationalTorus` represents `Σ_m d_m(𝖖^m v)·u^m` on the plain
generators, `FRationalTorus` represents `Σ_m f_m(𝖖^m v)·U_m` on the **dressed**
atoms, where the whole content of the dressing enters through
`U_a U_b = CC_{a,b}(v)·U_{a+b}`.

Only the *bare* residuals are stored, as `v^e` times a rational function of the
**root** characters. That split is load-bearing: a shift sends
`v^α ↦ 𝖖^{⟨c,α⟩}v^α` with `⟨c,α⟩` always an integer, so the whole fractional
content sits in one scalar per residual, and in the product law two such scalars
combine into the Dirac pairing — an integer exactly by admissibility. Without it
the frame would not close over `Z[𝖖, 𝖖⁻¹]` at a correlated form, where
`⟨ω^∨, ω⟩ = ½`. And the algebra depends on the **lattice**, not just the datum: a
residual legal in one form is refused at construction in another.

**Every admitted line is carried, odd height included.** A charge of odd
`⟨Σ⁺, m⟩` (where `Σ⁺ = Σ_{α>0} α`, so the Weyl vector is `½Σ⁺`) has a
half-integral atom phase `−½⟨Σ⁺,m⟩`. That is real, but the algebra never needs
the phase as a *number* — only through its coboundary `δS̃`, which is an
**integer even where the phase is not**, because the parity character
`π = ⟨Σ⁺,·⟩ mod 2` is Weyl-invariant *and* additive, so the halves cancel. The
cocycle restores the honest phase through an integral factor, and those charges
build. **No `𝖖^{1/2}` and no `i` appear anywhere** — the scalar ring stays
`Z[𝖖, 𝖖⁻¹]`. Certified at `SO(3)` against an independent standard-`Z²` BPS
chart (`H_0² = L_{(2,0)}` exactly, `I(H_0,H_0) = 1 − 𝖖² + 𝖖⁴ + 𝖖⁶ − 𝖖⁸`) and
orthonormal at `SO(5)` / `SO(7)`. What the centre character now decides is
*which forms need that correction to fire*, not which lines are missing.

### K. The Schur pairing in vacuum-state form

`aux_space.AuxSpace` — the pairing re-founded as `I_{a,b} = ⟨L_a·1, L_b·1⟩` on
the rational quantum torus, with `ρ` **never constructed**: the Hermitian slot
plus the two vacuum dressings generate the twist. Abelian-compatible by
construction, so the non-abelian case is the same structure with a nontrivial
measure rather than a new mechanism.

*Half-indices and 3d indices are the same pairing with one or both slots
dressed, but that machinery is **not** part of this release — it is Step 8.
What ships here is `I_{a,b}`, a 4d quantity.*

### M. The independent oracle for the odd-height claim

`pure_so3.PureSO3KAlgebra` — pure `SO(3) = PSU(2)` presented as a **BPS-quiver
chart** (nodes `(2,0), (−2,1)`, Dirac pairing `⟨n₀,n₁⟩ = 2`). It ships for one
reason: it is the check on this step's most load-bearing claim, and it is a check
the tier cannot perform on itself.

The claim is that a cocharacter of **odd** `⟨Σ⁺, m⟩` builds. Such a charge has a
half-integral atom phase `−½⟨Σ⁺,m⟩`, and an earlier reading concluded it therefore
needed a `𝖖^{1/2}` — outside the scalar ring `Z[𝖖, 𝖖⁻¹]`. The resolution is that
the phase is only ever needed through its coboundary `δS̃`, which is an integer even
where the phase is not, so the cocycle restores it. Certifying that *within* the
abelianized tier is weak evidence: the correction under test lives in the same
cocycle the tier's own pairing is computed from.

On the BPS chart the quiver nodes **are** the charge basis and the atom phase never
enters, so the two presentations share no machinery on the quantity compared. The
spinorial line `H_0 = F_{(1,0)}` — no `SU(2)` preimage, `⟨Σ⁺, ω^∨⟩ = 1`, the
smallest odd height there is — gives the same answer both ways:

    H_0²         = L_{(2,0)}   exactly (the SU(2) adjoint monopole), both sides
    I(H_0, H_0)  = 1 − 𝖖² + 𝖖⁴ + 𝖖⁶ − 𝖖⁸ + O(𝖖⁹),  term for term, both sides

and `H_0` is bar-invariant on the abelianized side. `tests/test_gn_flows.py`
(`test_H_odd_height_against_bps_oracle`) runs it. This is also the one module of
the tier that imports the Step-4 spine — unavoidably, the oracle *is* a
`BPSKAlgebra`.

## What this step updates in earlier layers

**A step is a version increment in the release history, not a frozen layer.** A
later step may — and routinely does — update an earlier step's modules in place;
that is what makes the release a sequence of updates rather than a stack of
sealed tiers.

| layer | module | why |
|---|---|---|
| Step 1 | `zplus_ring` | `UNZPlusRing` and the two ring homs the type-A seam needs |
| Step 1 | `laurent_poly` | **two correctness fixes** — below |
| Step 1 | `kalgebra` | a **named single-occurrence patch**, not a bump — below |
| Step 5 | `abe_kalgebra` | the datum-general torus selection, and the persistable chart memo |
| Step 5 | `root_datum` | the general-`G` factories (`so_n`, `sp_n`, `b_n_simply_connected`, `g_2`), the derived `ρ`-twist sign, and an inner-loop rewrite |
| Step 5 | `wrq_torus`, `matter_wrq_torus` | the **primitive cocycle** `CC` / `CC[N]`, weight-indexed rungs — what makes matter a *representation* — and the label-level `ρ` closed form |
| Step 5 | `weyl_torus_ring`, `quiver_wrq_torus`, `urq_torus` | follow the substrate |
| Step 5 | `pure_su2_kalgebra`, `pure_sun_kalgebra` | the `SO(3)` frame |
| Step 5 | `un_nf_kalgebra`, `su2_unf_abe_kalgebra` | **a correction**: these were computing the trace at the chart's default window, which silently corrupts every `𝖖`-order above `𝖖⁴`. They now pass a window matched to the requested order. |

**`laurent_poly` — two defects that shipped.** A non-integral coefficient was
*silently truncated*: the zero-test read the original `c` while the store kept
`int(c)`, so `Fraction(1,2)` passed the test and landed as **0**, and `3/2` as
`1`. The stored zero then broke the class's documented invariant (a sparse dict
of *non-zero* coefficients), and `__add__`'s two paths disagreed about it — the
general path raised `KeyError` where the monomial fast path did not, so the same
sum crashed or not depending on how many terms the right operand happened to
have. A non-integral coefficient now raises; the two paths now agree.

**`kalgebra` — patched, not bumped.** The published contract deliberately drops
`structure_constants`, `_sc_cache_dict`, `cache_identity` and the
structure-constant cache's JSON (de)serialisation — the engine-coupled surface —
so a wholesale bump would silently re-add all of it. The one genuine improvement
is ported instead, as an exact single-occurrence substitution:
`verify_inner_product_consistent` stopped at the *first* base that overrides
`inner_product`, so an intermediate base which honestly raises
`NotImplementedError` on a given realisation suppressed the check entirely; it
now walks past such a base to the root definition, which always computes. **Port
the change, not the file** is the rule those two rows illustrate.

`root_data` — the gauge-data leaf that the general-`G` factories reach through a
lazy import — moved into this layer, since it is gauge data and Step 7 is what
needs it.

## Self-test

    python run_tests.py          # the whole gate, this layer included

The Step-7 block builds general-`G` charts; asserts that with `optimizations=()`
**every** label — Wilson, minuscule and bubbled alike — is built by the axiom
route, so the one-route claim is runnable rather than merely stated; checks that
the two declared optimizations return the *identical* element; runs the five
axioms as a checklist with their **negative controls** (a bubbling-stripped
canonical rejected by (★) alone, a wrong `e` by W2, `L_1·L_1` asked as `L_1` by
support + W2 + `O(𝖖)`); asserts the odd-height charges **build, certify and are
orthonormal**, against an independent BPS oracle; exercises the three `su(2)`
global forms and the line lattice's tori; and runs the contract verifiers through
the universal `KAlgebra` surface.
