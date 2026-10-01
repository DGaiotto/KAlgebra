# Residue cancellations and the conventional presentation of `AbeKAlgebra` elements

**Charter (user, 2026-07-27).**  *"After much effort, the 'easy-Neumann'
conventions for U(N) half-indices and 3d indices seem to make sense.  They give
a specific action of `L_{m,e}` and `L̃_{m,e}` as difference operators which
should give two canonical ways to identify `U_m` with (something like
`ψ_m(𝖖^m v)`)`u^m` with `u, v` conventional quantum torus generators.  Let me
call these **conventional presentations** of the `AbeKAlgebra` elements.
Discussing with mathematician friends, I was told that the `AbeKAlgebra`
elements can be recognized by certain **residue cancellations**.  I have not yet
interiorized the statement.  A related fact is that
`L_{m,e}|N] = (combination of L_{0,e'}'s)|N]` and similarly for `L̃`.  This
requires a cancellation of denominators in the conventional presentation when
acting on the '1' wavefunction.  Analogous cancellations must happen for `N(k)`
and also for Neumann enriched by chiral multiplets in any representation of the
gauge group.  I would like to explore the implications of that, especially for
groups for which we do **not** yet know the correct bubbling contributions.
Long term investigation, lets start accumulating facts."*

**Status.**  Session 1 (2026-07-27) — R1–R7 below, all exact, all green.  Fact accumulation, no contract surface
touched.  Battery: `experiments/residue_cancellation_conventional.py` [retired 2026-09-19 with the type-A keystone; the measurement stands as recorded]
(exact throughout; `--full` adds the reconstruction pass).

**Terminology.**  "Conventional presentation" is the **user's name**
(2026-07-27) for writing an `AbeKAlgebra` element as an honest difference
operator in conventional quantum-torus generators `u, v` — the atom dressing
`ψ` absorbed.  It is used here in exactly that sense and nothing else is
coined.  Everything else below is repo vocabulary (`URQTorus`, `U_m`, `ψ_m`,
residual `f_m`, the wall, bubbling, W1/W2 acceptance).

> **False-friends flag (CLAUDE.md discipline).**  The criterion measured here
> is *close* to the pole/residue characterisation of the Coulomb-branch algebra
> inside its abelianisation in the BFN/BDG literature — which Plan 30 ruling D2
> pins as **related and non-load-bearing**.  Nothing below is imported from
> there: every statement is a measurement against the repo's own
> `PureUNKAlgebra` registry, in repo conventions.  The resemblance is recorded
> so a later session does not mistake agreement for derivation.

---

## 1. The conventional presentation, concretely

On the enriched rational quantum torus (`pun_enriched_qt.md` §1) an element is
stored in the f-presentation `x = Σ_m f_m(𝖖^m v)·U_m` on the atoms
`U_m = u^m·ψ_m(v)`.  Expanding the atoms gives the conventional presentation

    D  =  Σ_a  d_a(v) · u^a ,        d_a = T_a(f_a) · ψ_a ,

which is exactly `URQTorus.to_chart()` (a `DOp`).  Because `u^a f(v) = f(𝖖^{2a}v)u^a`,
`D` acts on functions of `v` by

    (D·f)(v)  =  Σ_a  d_a(v) · f(𝖖^{2a} v).

`ψ_a` carries the W-boson poles `∏_{i<j}∏_{l=0}^{a_i-a_j-1}(v_i − 𝖖^{−2l}v_j)`
in its denominator, so **each individual `d_a` is singular on the walls** even
though the operator is not.

**The two presentations.**  Left multiplication is the `u` operator, right
multiplication the dual `ũ` (`pun_enriched_qt.md` §6e; `L̃_a = ρ(L_a)^†`).  In
this session the second presentation is exercised through the `ρ`-image
(`URQTorus.rho()`), which is the repo's realisation of that duality — check R5
below.  A direct test of the `ũ`-frame *identification* `U_m ↔ ψ_m(𝖖^m v)u^m`
(as opposed to its `ρ`-image proxy) is **open**.

## 1b. The map `U_m ↦ difference operator` for GENERAL `G` (user question, 2026-07-27)

**We have it, and it is canonical.**  `ψ_m` is `wrq_torus.dressing_psi(datum, m)`,
defined for any `RootDatum`.  At dominant `m` (Weyl-transported elsewhere):

    U_m  =  u^m · ψ_m(v),
    ψ_m  =  ε · 𝖖^{Q(m)−S(m)} · v^{W(m)} · ∏_{α>0} ∏_{l=0}^{⟨α,m⟩−1} (1 − 𝖖^{2l} v^α)^{−1}
    W(m) = Σ_{α>0}⟨α,m⟩·α ,  Q(m) = Σ_{α>0}⟨α,m⟩(⟨α,m⟩−1) ,  ε = (−1)^{Σ_{α>0}⟨α,m⟩ + S}

with `S = RootDatum.atom_phase(m)`.  Everything is roots + the cocharacter
pairing — no type-A input.  The pole divisor is the W-boson ladder of `m`: one
simple factor per (positive root, level `l < ⟨α,m⟩`).  That divisor is the whole
geometric content of the map.

**The atom-phase freedom is not a freedom of the map.**  `S` is the only scalar
freedom in `U_m` (`U_m ↦ ±𝖖^{S(m)}U_m`, = the choice of cocycle representative),
and it **cancels out of the difference operator**: rescaling `U_m` forces
`f_m ↦ 𝖖^{−S}f_m` on the residual, so `d_m = T_m(f_m)·ψ_m` is unchanged.  So the
map to difference operators is canonical; the freedom lives only in how one
splits an element into (atom) × (residual).  (Corroborated: perturbing
`atom_phase` by linear and quadratic functions of `m` leaves the pole divisor
identical at SU(2) and U(2).)  `S` is independently constrained by the repo's
**D10 bar-honesty ruling**: its non-linear content must be `−⟨ρ, m⟩ =
−½Σ_{α>0}⟨α,m⟩`, with linear pieces free (coboundaries that cancel in the
cocycle).

**Beyond type A: measured.**  `RootDatum`'s *factories* are type-A (`u_n`,
`su_n`, `torus`, `product_datum`), but the constructor accepts explicit positive
roots, so `B2 = SO(5)` is buildable by hand (ω-basis; Cartan `[[2,−2],[−1,2]]`;
`Φ⁺ = {(2,−2), (−1,2), (1,0), (0,2)}`; `|W| = 8`, `ρ = (1,1)`).  On it:

* `dressing_psi` evaluates at every `m` tried, with the expected ladder divisor;
* the `atom_phase` parity guard passes (`⟨Σ⁺,m⟩ = 2(m₁+m₂)`, always even);
* the cocycle `U_aU_b = R_{a,b}U_{a+b}` evaluates on 25 pairs, 0 failures;
* coroot extraction (from the Weyl reflection) gives `⟨α, α^∨⟩ = 2` on all four
  roots, short and long — so the affine-reflection partner rule of R7 is
  well-defined in the non-simply-laced case.

**What is missing beyond type A is the CANONICALS, not the map.**  B2 in the
coroot lattice has **no minuscule cocharacter** (swept `|mᵢ| ≤ 3`: none;
`wrq_torus.is_minuscule` agrees) — *every* monopole bubbles, so the type-A
dressed-minuscule strategy has nothing to start from.  That makes B2 the minimal
clean testbed for Plan 24, and the natural target for the R4 reconstruction.

## 2. The criterion, stated

Let `D = Σ_a d_a(v) u^a`.  On the wall `v_i = 𝖖^{2l} v_j` the shifted arguments
of the term `a` are `(𝖖^{2a}v)_i = 𝖖^{2a_i+2l}v_j`, `(𝖖^{2a}v)_j = 𝖖^{2a_j}v_j`.
A **Weyl-invariant** test function therefore takes the *same* value at `a` and at
its **wall partner**

    s^{(l)}_{ij}(a):     a_i ↦ a_j − l ,    a_j ↦ a_i + l ,    (other slots fixed)

and at no other term.  **Derivation** (asked for by the user, 2026-07-27; this
was derived from scratch and only then found to match the reported statement).
On the wall the reflection acts on the *point* as `s_α(v) = 𝖖^{−2l α^∨}·v`
(because `s_α(v)^λ = v^λ (v^α)^{−⟨λ,α^∨⟩}`), hence

    s_α(p_a) = 𝖖^{2 s_α(a)}·s_α(v) = 𝖖^{2(s_α(a) − l α^∨)}·v = p_b ,
    b = s_α(a) − l·α^∨ ,

exactly, in every component — the affine Weyl reflection fixing `⟨a,α⟩ = −l`.
So the polar part of `D·f` at the wall is `Σ_pairs (Res d_a + Res d_b)·f(p_a)`,
and regularity of `D` on symmetric functions is therefore equivalent to the
**pairwise residue cancellation**

    Res_{v_i = 𝖖^{2l} v_j}  [ d_a  +  d_{s^{(l)}_{ij}(a)} ]  =  0        (★)

for every wall and every pair.  `s^{(l)}_{ij}` is an involution, so the terms
genuinely pair up; nothing is left over.

This is the executable form of "recognized by residue cancellations".  It is
tested in the battery by exact division (does the pair sum still carry the wall
factor in its denominator?), never numerically.

**Relation to the `|N]` condition — (★) IS the charter's statement, read on the
MODULE (user, 2026-07-27: "of course you are right: `L_{m,e}` must be able to act
nicely on `L_{0,e'}|N]`").**

The two are equivalent, and the earlier "strictly weaker" reading in this note
applied to the wrong object — the single vector `|N]`, not the module it
generates.  The correction:

* `|N]` is **cyclic for the Wilson subalgebra**.  The `m = 0` canonical is the
  single atom `U_0` with residual `χ_e`, so `L_{0,e}|N] = χ_e` **exactly**, for
  every `e` (measured: 12/12 Wilson labels give precisely `1·χ_e`).  Hence

      span{ L_{0,e'}|N] }  =  Λ  =  the symmetric Laurent polynomials  =  R(G).

* Therefore *"`L_{m,e}` acts nicely on `L_{0,e'}|N]` for all `e'`"* is exactly
  *"`L` preserves `Λ`"*, which is exactly (★).  Nothing is lost:

      (★)   ⟺   `L` acts on the Neumann module `A·|N] = R(G)`.

* The weaker sum-over-pairs reading is what you get by testing the **single**
  vector `f = 1`: there all the values `f(p_a)` coincide, so only the total
  `Σ_pairs (Res d_a + Res d_b)` need vanish.  Testing the whole module (or,
  equivalently, the `{|N(k)]}` tower) separates the pairs and recovers (★).
  The module side is measured, not argued — R4b, 562/562, plus R8's explicit
  matrix elements.

**Why this is the right way to say it.**  It makes the criterion
*presentation-free*: an `AbeKAlgebra` element is precisely a `𝖖`-difference
operator preserving the representation ring, with `|N] = 1 ∈ R(G)`, Wilson lines
acting by multiplication by characters, and monopoles acting triangularly.  That
is the **polynomial (Macdonald-style) representation** — which
`aux_neumann_wavefunction.md` §2l Task F named from the pole side; here it has a
reason rather than a resemblance.  Measured structure at U(2) (R8):

| operator | action on `Λ` |
|---|---|
| `L_{(1,1),(0,0)}` — det monopole | **diagonal**, `χ_λ ↦ −𝖖^{2Σλ−1}·χ_λ` |
| `L_{(1,0),(0,0)}` — basic 't Hooft | **triangular**, leading eigenvalue `𝖖^{2λ₁}`; e.g. `χ_{(2,0)} ↦ (−1+𝖖²)χ_{(1,1)} + 𝖖⁴χ_{(2,0)}` |
| `L_{(1,0),(0,1)}` — the `\|N]`-annihilating dyon | **strictly raising**: `χ_{(0,0)} ↦ 0`, `χ_{(1,0)} ↦ (−1+𝖖²)χ_{(1,1)}` |

For the group-general programme this is the useful form: at any `G` the target is
"the `𝖖`-difference operators preserving `R(G)`", with no reference to a chart,
a chamber, or a bubbling formula.

## 3. Measured facts (pure U(N), N = 2 and 3, whole registry)

`PureUNKAlgebra.cached(2)` = 25 labels, `cached(3)` = 39 labels.  All exact.

### R1 — the Neumann row

`|N]` is the constant wavefunction, so `L·|N] = Σ_a d_a(v)`.  **Measured, 64/64:**
the sum is always a Laurent polynomial — every W-boson denominator cancels — and,
sharper than the charter asks, it is always a **single** `U(N)` character with a
unit monomial coefficient (or zero).  Since `L_{0,e}·|N] = χ_e`, the row reads
directly in the `L_{0,e'}|N]` basis.  Sample (U(2)):

| label | `L·\|N]` | label | `L·\|N]` |
|---|---|---|---|
| `((0,0),e)` | `χ_e` | `((2,0),(0,0))` | `χ_(0,0)` |
| `((1,0),(0,0))` | `χ_(0,0)` | `((2,1),(0,0))` | `−𝖖⁻¹χ_(0,0)` |
| `((0,−1),(0,0))` | `−𝖖·χ_(0,0)` | `((1,−1),(0,0))` | `−𝖖·χ_(0,0)` |
| `((p,p),(0,0))` | `(−𝖖⁻¹)^p` | `((1,1),(1,0))` | `−χ_(1,0)` |
| `((1,0),(0,1))`, `((1,0),(−1,0))`, `((0,−1),(0,±1))` | `0` | `((0,−2),(0,0))` | `𝖖²χ_(0,0)` |

Two readings worth recording:

* the **central tower** `L_{(p,…,p),0}·|N] = (−𝖖⁻¹)^p` at U(2) (`(−𝖖⁻³)^p`-shaped
  at U(3): `−𝖖⁻³, 𝖖⁻⁶` for `p = 1, 2`) — the wavefunction is flat along the
  U(1) centre, matching `aux_neumann_wavefunction.md` §2l Task A
  (`f_{m+(1,1)} = f_m`);
* the **annihilations** `L_{(1,0),(0,1)}·|N] = 0` etc. are the repo's already-recorded
  bare-'t Hooft annihilation phenomenon (§2b–2c) seen in this frame.

### R2 — the criterion holds, and is discriminating

(★) holds for every one of the 64 canonicals.  Negative controls (U(2)):

| element | R1 pole-free | R2 |
|---|---|---|
| bare atom `U_(1,0)` | ✗ | ✗ |
| `L_{((1,0),(0,0))}`, one sector × `𝖖²` | ✗ | ✗ |
| `L_{((1,0),(0,0))}`, one sector × `2` | ✗ | ✗ |
| `L_{((2,0),(0,0))}`, **bubbling sector deleted** | ✗ | ✗ |
| `L_{((2,0),(0,0))}`, **bubbling sector × `𝖖²`** | ✗ | ✗ |

(`U_(1,0)+U_(0,1)` also passes — correctly: at difference 1 there is no bubbling
and that sum *is* the canonical.  It is not a false positive.)

### R3 — the pole-location law

**Measured on all 64:** `d_a` has a **simple** pole at wall `k` **iff** the wall
partner `s_k(a)` lies in the support and differs from `a`.  Poles are never
higher order, and never occur where the partner is absent or self-paired.  This
gives the denominator of an unknown sector for free once its support is known.

### R4 — bubbling from cancellation (`--full`)

Keep only the **extremal Weyl orbit** of a canonical (the leading Levi
character, no bubbling) and treat every interior sector as unknown, with its
denominator fixed by R3 and its numerator a `Stab_W(a)`-symmetric homogeneous
ansatz of degree = number of walls.  The conditions (★) against the known
sectors are a linear system; solved exactly over `Q` at `𝖖 ∈ {3/2, 5/3, 7/4}`.

**Measured, every bubbling label at N = 2, 3:** the system is **never
inconsistent**, and its affine solution set **always contains the true
canonical**.  The null directions are numerators divisible by the **full wall
product** — i.e. *constant* residuals on the interior atoms, which are exactly
**lower canonicals**.

The `d = 2` 't Hooft makes it visible.  In the conventional presentation

```
L_{(2,0),(0,0)} =   𝖖²v₁²        /[(v₀−𝖖²v₁)(v₀−v₁)]      · u^(0,2)
                + (−𝖖⁻²−1)v₀v₁ /[(v₀−𝖖²v₁)(v₀−𝖖⁻²v₁)]   · u^(1,1)   ← bubbling
                +    v₀²        /[(v₀−𝖖⁻²v₁)(v₀−v₁)]      · u^(2,0)
```

At `v₀ = 𝖖²v₁` the `u^(0,2)` term has residue `𝖖²v₁/(𝖖²−1)` and the bubbling
term exactly `−𝖖²v₁/(𝖖²−1)`.  The two wall conditions on the bubbling numerator
have **determinant zero**, with null direction

    (v₀² + v₁²) − (𝖖² + 𝖖⁻²)v₀v₁  =  (v₀ − 𝖖²v₁)(v₀ − 𝖖⁻²v₁),

i.e. adding `λ·U_(1,1)` with a constant residual — the det-monopole, a lower
canonical.  So:

> **The shape of the statement.**  Residue cancellation determines the bubbling
> **modulo the span of lower canonicals**.  The residual finite-dimensional
> ambiguity is the Kazhdan–Lusztig one, and it is exactly what bar-invariance +
> the `O(𝖖)` leading condition — the repo's W1/W2 `certify_canonical` — already
> fixes.  Residue cancellation supplies *membership in the algebra*; W1/W2
> supplies *the canonical representative*.

Nullity is 1 where there is a single interior sector, 4 where there are three
(because the battery solves sector-by-sector and drops the conditions coupling
two unknown sectors — tightening that is an obvious next increment).

### R4b — general symmetric test functions

(★) was derived for an *arbitrary* Weyl-invariant test function, not for the
constant.  **Measured, 562/562** (`D·χ_λ` for a spanning set of `λ`, over the
whole registry at N = 2 and 3): every canonical maps symmetric Laurent
polynomials to symmetric Laurent polynomials.  That is the full membership
statement — the operator preserves the ring of `W`-invariants — rather than a
one-state accident.

### R6 — `N(k)`: the criterion DERIVES the Chern–Simons Gaussian

The charter asks for the analogous cancellation for `N(k)`.  In the easy-Neumann
frame the CS states are the bare monomials `v^{km}`
(`aux_neumann_wavefunction.md` §2l), which in the conventional presentation
twists the operator to `D^{(k)} = Σ_a v^{ka} d_a(v) u^a`.

**Measured: that naive twist FAILS**, at exactly the bubbling labels
(`|m_i − m_j| ≥ 2`) — 22/25 at U(2), 30/39 at U(3), for every `k` tried.
The reason is visible in (★): at the wall `v_i = 𝖖^{2l}v_j` the sector monomials
of `a` and `s^{(l)}_{ij}(a)` restrict to `𝖖^{2lka_i}` versus `𝖖^{2lk(a_j−l)}`
times the same `v`-monomial — equal only when `l = 0`.  Demanding equality,

    c(a) − c(s(a))  =  2lk(a_j − l − a_i)      ⟹      c(a)  =  k·Σ_i a_i²

(a quadratic, hence a **Chern–Simons coupling**).  With the Gaussian,

    |N(k)|   ↔   𝖖^{k·Σ_i a_i²} · v^{k a}

R1 and R2 are restored **100%** — 25/25 and 39/39 at `k = 1, 2, 3, −1`.  Two
sharpenings:

* `Σ a_i²` and `Σ a_i(a_i−1)` work equally; they differ by the linear,
  Weyl-invariant `tr(a)` — a framing convention.  This reproduces at general
  `k` and general `N` what `aux_neumann_wavefunction.md` §2b measured only as an
  SU(2) *sign*: *"the two quadratic variants `(−1)^{m(m∓1)/2}` both work (they
  differ by the linear `(−1)^m` = a CS convention)"*.
* the opposite sign `𝖖^{−kΣa_i²}` **fails** — the criterion fixes the Gaussian's
  sign relative to the orientation of `v^{ka}`.

**What the Gaussian actually IS — a frame artifact, and it IS in [KW] (user
question, 2026-07-27; supersedes the first reading of this section).**  The
`f`/wavefunction frame stores the residual as a function of the *half-shifted*
argument `ζ = 𝖖^m v`, so the CS state written there is the **bare monomial**
`ζ^{km}` — literally [KW]'s `z^{κ(m)}` (`threed/CONVENTIONS.md` §"Chern–Simons":
`e^{−S_CS} = ∏_j z_j^{k s_j}`, a bare holonomy monomial with no explicit
𝖖-Gaussian).  Pushing it to the bare-`v` conventional presentation applies the
half-shift `T_m : v_i ↦ 𝖖^{m_i}v_i`:

    T_m( ζ^{km} )  =  𝖖^{k·Σ_i m_i²} · v^{km} .

**Verified exactly, 100/100** (whole U(2) registry, `k = 1, 2, 3, −1`): twisting
the stored residual by the bare `v^{km}` gives the same operator as twisting the
conventional presentation by the Gaussian.  So the criterion did not *discover* a
Chern–Simons coupling — it detected that the state had been written in the wrong
frame, and the "derivation" recovers the frame conversion.  It remains a genuine
check (it fixes the sign and forbids non-linear deviations), but it is frame
bookkeeping, not new physics.  The leftover `Σa_i(a_i−1)` vs `Σa_i²` freedom is
correspondingly a different choice of frame variable — the same linear-in-`m`
half-level/framing shift visible in `CONVENTIONS.md` as [DGG]'s `𝖖^{m/4}` at
flavour CS level `−1/2`.

At `k ≠ 0` the rows become genuine **multi-character** combinations already in
the pure theory — the charter's `Σ_{e'} L_{0,e'}|N(k)]` in full:

    L_{(2,0),(0,0)}  · |N(1)]  =   𝖖²·χ_(1,1)  +  𝖖⁴·χ_(2,0)
    L_{(1,−1),(0,0)} · |N(1)]  =  −𝖖·χ_(0,0)   −  𝖖³·χ_(1,−1)
    L_{(0,−2),(0,0)} · |N(1)]  =   𝖖⁴·χ_(−1,−1) + 𝖖⁶·χ_(0,−2)

### R5 — the other presentation

The same three checks on the `ρ`-image (`L̃_a = ρ(L_a)^†`, `pun_enriched_qt.md`
§6e) pass on all 64.  The registry is **not** `ρ`-closed, so this is a genuine
extra test rather than a relabelling of the sweep.

### R7 — GROUP-GENERAL: the criterion is affine-Weyl-invariance of the poles

There is no DOp bridge on the group-general tier (ruling D4), so the
conventional presentation is built from `dressing_psi` directly:
`d_m = q_shift(f_m, m)·ψ_m`.  `TorusRational` denominators are keyed `(α, k)`
for `1 − 𝖖^k v^α`, i.e. the wall `v^α = 𝖖^{−k} = 𝖖^{2l}`, `l = −k/2`.  Redoing
the derivation of §2 on a general root datum, the wall partner is the **affine
Weyl reflection**

    s_{α,l}(a)  =  s_α(a) − l·α^∨  =  a − (⟨a,α⟩ + l)·α^∨ ,

which reduces to the type-A rule.  So (★) says exactly: **the pole structure of
the conventional presentation is affine-Weyl-invariant.**  Nothing in it is
type-A specific.

**Measured:**

| datum | labels | result |
|---|---|---|
| U(2) (cross-check through the general-G code path) | 5 | all pass — agrees with the type-A sweep |
| pure SU(2), `PureSU2KAlgebra` (fully native) | `m = 0…4`, `e = 0…2` (15) | **all pass**, full bubbling towers (support up to 9) |
| pure SU(3), `PureSUNKAlgebra` (oracle-backed) | 8, incl. the bubbling adjoint monopole `(−1,0,1)` and `(−2,0,2)` | **all pass** (support up to 19) |

SU(2) is the discriminating case: `⟨m,α⟩ = 2m`, so the **basic** 't Hooft already
bubbles (`aux_vacuum_pairing.md` §4g).  And the criterion earns its keep there:

> **It rejected a wrong element in the wild.**  Run on
> `wrq_torus.build_canonical(su_2(), (1,), (0,))`, (★) **fails** — the wall
> partner at `k = ±2` is the `m = 0` sector, which that builder omits.  This is
> correct and independently documented: *"the generic
> `wrq_torus.build_canonical` is cone/minuscule-only and does not reach the
> bubbling adjoint"* (`implementations/pure_sun_kalgebra.py` docstring).  The
> genuine `PureSU2KAlgebra` canonical at the same label passes.  So the
> criterion distinguishes a real canonical from an un-peeled seed **without
> knowing the bubbling formula** — which is precisely the charter's target.

### Matter — first look (U(N)+N_f)

`UNNfKAlgebra(N, N_f)` for `(N,N_f) ∈ {(1,1),(2,1),(1,2),(2,2)}`, per flavour
level (`MatterURQTorus.to_family()`): R1, R2, R3 hold at every label and level
tried.  Here the Neumann row is a **genuine multi-character combination** —
the charter's `Σ_{e'} L_{0,e'}|N]` in full — e.g. at U(2)+N_f=1,

    L_{((-2,-2),(0,0))} · |N] ,  flavour level 2 :
        (𝖖⁻⁴+𝖖⁻²+1)·χ_(1,1)  +  (𝖖⁻²)·χ_(2,0)

**Honest limitation:** `to_family()` de-dresses the matter rungs, so this probe
tests the **gauge** walls level by level, *not* the chiral-multiplet poles.  The
genuine matter cancellation — the one the charter asks about for "chiral
multiplets in any representation" — lives in a frame where the matter
denominators are visible; the natural counterpart is the derived per-cell window
`Ξ` of `aux_vacuum_pairing.md` §4m (trivial at `⟨m,w⟩ ≥ 0`, the μ-dressed double
window = the paper's `u_±` numerators at `⟨m,w⟩ < 0`).  Doing this properly is
the top matter-side next step.

## 4. Where this already sits in the repo

This is the **same phenomenon** as `aux_neumann_wavefunction.md` §2l **Task F**
("Easy-Neumann is a POLYNOMIAL (Macdonald-style) representation": the bubbling
poles of a canonical chart sit at exactly the first `K−k` zeros of each
Pochhammer wing of `1/f_k`, all simple, with an exact cancellation lemma), seen
from the operator side rather than the state side.  Task F's own caveat —
*"a measured pattern plus the exact lemma, not a general proof — that needs the
closed-form bubbling pole set for arbitrary `(m,λ)`"* — is precisely what (★)
sidesteps: (★) does not need the pole set in advance, it **derives** the polar
part from the extremal orbit.

## 4b. Can we recover SU(2) with no U(2) oracle?  (user, 2026-07-27)

**Question, in the user's sharpened form:** *"is `L_{m,e}` determined uniquely by
the leading+bubbling structure, the `O(𝖖)` condition on bubbling, bar invariance
(all conditions on the `U_m` form of the operator) plus the good action on
symmetric functions?"*  And the observation that motivates it: *"it naturally
uses both the `Σ_m a_m(𝖖^m v)U_m` form of the element and the map to difference
operators."*  Exactly so — (i)–(iii) live on the `U_m` form, (iv) = (★) lives on
the difference-operator image, and the two constrain each other.

**Answer, measured at `m = 1`: YES, uniquely — and every ingredient is
load-bearing.**  Battery: `experiments/su2_bubbling_from_residues.py`.

Setup: pretend SU(2) has no U(2) oracle and no known adjoint fibre.  Allowed
input — `su_2()`, `dressing_psi`, `leading_orbit` (verified provenance: only
`levi_character` + the Weyl action, no U(N) anywhere; its own docstring calls it
*"the bar-non-invariant seed bubbling-finding dresses into the canonical"*), the
doubly-tropical interval as support, and the axioms.  **Held out: every interior
sector.**  `PureSU2KAlgebra` is used only to grade the answer.

**Step 1 — (★) forces the interior numerator on every wall, exactly.**  By exact
division in `Z[𝖖^±]`, and correct in every case checked: `m = 1` at `e = 0,1,2`,
and `m = 2` (all six wall values that couple an unknown to a *known* sector).

**Step 2 — the ingredients, and what each one actually does.**  Measured, and
more interesting than expected:

| condition | what it rejects | what it MISSES |
|---|---|---|
| W1+W2 (`WRQTorus.well_formed`) | mis-shaped leading data | **the bare seed passes it** (unit residuals are palindromic; the leading orbit is fine) — and so does any `λ·L_{0,j}` admixture with `λ` palindromic, since the wall product is *itself* palindromic and the leading orbit sits at `m = ±1` |
| (★) | the bare seed (= the R7 negative control), and any wrong polar part | the lower-canonical freedom — it fixes the bubbling only *modulo* `span{L_{0,j}}` |
| KL `O(𝖖)` on the bubbling | the lower-canonical admixture | — |

So **neither the `U_m`-side conditions nor (★) suffices alone**, and they fail in
*complementary* directions.  Together they are rigid.

**Step 3 — the concrete avatar of the `O(𝖖)` condition.**  Adding `L_{0,j} =
χ_j·U_0` adds a *Laurent polynomial* to `f_0`.  KL says its coefficient lies in
`𝖖Z[𝖖]`, and bar-invariance then forces it to vanish.  Equivalently and
executably: **`f_m` must be a PROPER rational function** — no polynomial part,
i.e. decaying as `v → 0` and `v → ∞`.  With `nw` walls the wall product spans
`v^0 … v^{2nw}`, so properness reads `1 ≤ deg(C) ≤ 2nw − 1`.  (This is the same
"no regular remainder" observation that made R4's U(2) `d=2` partial fractions
close.)  Note W2 as implemented does *not* imply it — W2 inspects the q-extreme
slice of the *leading* orbit only.

**Result.**  With (★) + W1 + properness, the window scan returns a **unique**
interior numerator, equal to the truth:

| label | recovered | true |
|---|---|---|
| `m=1, e=0` | `(𝖖⁻¹+𝖖)·v²` | `(𝖖⁻¹+𝖖)·v²` |
| `m=1, e=1` | `v + v³` | `v + v³` |
| `m=1, e=2` | `(𝖖⁻¹+𝖖)·v²` | `(𝖖⁻¹+𝖖)·v²` |

i.e. `f_0 = (𝖖⁻¹+𝖖)v²/[(1−𝖖⁻²v²)(1−𝖖²v²)]` — **the adjoint fibre**, which is
precisely the "single structural input" that `PureSU2KAlgebra` (#765) takes as
given.  It is recoverable from the axioms.

**Open (honest).**  At `m ≥ 2` two of the four walls per interior sector couple
*unknowns to each other*, so the one-pass known→unknown recursion does not close;
it needs a joint solve over the interior sectors.  Every known-partner wall value
at `m = 2` is exactly right, so nothing suggests a failure — but the joint solve
is not done, and uniqueness at `m ≥ 2` is therefore **not yet established**.
Also open (user, 2026-07-27): requiring a good action of **both** `L_{m,e}` and
`L̃_{m,e}` — plausibly a strictly stronger constraint on a *candidate* (a
non-canonical element could satisfy (★) while its `ρ`-image does not), and the
natural thing to try first on the coupled `m ≥ 2` system.

## 4c. General `G`: SU(3) and SO(5) recovered; G₂ in progress (user, 2026-07-27)

**Charter:** *"Lets aim for a joint solver working for all m in SU(2)"*, then
*"look at SU(3), SO(5), G_2 in order."*  Solvers:
`experiments/su2_joint_bubbling_solver.py` (rank-1) and
`experiments/general_g_joint_bubbling_solver.py` [retired 2026-09-19 with the type-A keystone; the measurement stands as recorded] (datum-general).

Three things had to generalise from rank 1, and all three are **derived**:

* **support** — the doubly-tropical interval = lattice points of `conv(W·m)` in
  the coset `m + Q^∨`.  Checked against the `PureSUNKAlgebra` ground truth at
  SU(3): 4/4 exact.
* **wall restriction with torsion** — a wall `v^α = 𝖖^{−k}` with `α = g·α₀`
  determines `v^{α₀}` only up to a `g`-th root of unity, so the restriction
  splits into `g` isotypic components, each vanishing separately.  This is the
  rank-1 even/odd split generalised (SU(2): `α = (2)`, `g = 2`); invisible at
  SU(3) (all roots primitive); live again at B2/G₂.
* **proper window** — the numerator's Newton polytope must lie in the
  **relative** interior of the denominator's Newton zonotope `Σ_α [0, m_α·α]`,
  i.e. `p = Σ_α s_α α` with `0 < s_α < m_α`.  *Relative* matters: when all walls
  of a sector share one root the zonotope is a degenerate segment in a
  higher-dimensional lattice and the absolute interior is empty, while the true
  numerator is a single monomial sitting at `α` (measured at SU(3)).  Decided
  exactly by Fourier–Motzkin.

The derived `K_f(n) = (K_d(n) \ P_n) − ⟨n,α⟩` was re-measured at SU(3) and holds
on every sector of every label checked.

**Results.**  All exact, symbolic in `𝖖` (fraction-free Bareiss over `Z[𝖖^±]`).

| datum | `m` | support | reps | unknowns | equations | Bareiss | bar/(★)/palin. | `⟨L,L⟩(K=4)` | vs truth |
|---|---|---|---|---|---|---|---|---|---|
| SU(3) | (−1,−1) | 7 | 1 | 8 | 102 | unique | ✓ | `1 − 2𝖖² + 3𝖖⁴` | **MATCH** |
| SU(3) | (−1,−2) | 10 | 2 | 9 | 186 | unique | ✓ | `1 − 𝖖²` | **MATCH** (7/7) |
| SO(5) | (1,0) | 5 | 1 | 3 | 32 | unique | ✓ | `1 − 𝖖² + 𝖖⁴` | *no repo truth* |
| SO(5) | (0,1) | 9 | 2 | 11 | 292 | unique | ✓ | `1 − 𝖖²` | *no repo truth* |
| SO(5) | (1,1) | 5 | 1 | 3 | 32 | unique | ✓ | `1 − 𝖖² + 𝖖⁴` | *no repo truth* |
| **G₂** | (1,0) | 13 | 2 | 16 | 804 | unique | ✓ | `1 − 𝖖² + 𝖖⁴` | *no repo truth* |
| **G₂** | (0,1) | 7 | 1 | 5 | 114 | unique | ✓ | `1 − 𝖖² + 𝖖⁴` | *no repo truth* |

In every case the square system is solved symbolically and then **every** dropped
equation is verified against the answer (102/102, 186/186, 32/32, 292/292,
804/804, 114/114) — the row selection defers work, it does not assume anything.

**What made G₂ reachable: Weyl covariance.**  The canonical is a Weyl-invariant
element, so `f_{w(n)} = w·f_n` — MEASURED exactly on the true SU(2)/SU(3)
canonicals (216/216).  Unknowns therefore live only on **orbit representatives**,
with the representative's residual `Stab_W(n)`-invariant.  G₂ at `m = (1,0)` goes
from 139 unknowns to **16**, which is the difference between "does not finish"
and "seconds".  Two traps on the way, both measured rather than reasoned around:
the symmetrisation must be of the **rational function** (`TorusRational.weyl_act`
re-canonicalises negative roots with a monomial correction, so hand-rolling the
monomial transport gives 0 unknowns), and it must be an **orbit sum over distinct
images** (summing over all of `Stab` counts each orbit element `|Stab_λ|` times,
putting the true element in the span only with `1/|W|` coefficients, which the
integral solve rejects as inconsistent).

SO(5) and G₂ have **no repo ground truth** (it is not even a `RootDatum` factory — built
by hand from explicit positive roots), so the certificate there is
**orthonormality**: `⟨L,L⟩ = 1 + O(𝖖)`, a *trace* computation with no residue
input, hence genuinely independent of the conditions that produced the element.
Note also that B2 in the coroot lattice has **no minuscule cocharacter**, so
every one of these monopoles bubbles — there is nothing for the type-A
dressed-minuscule strategy to seed from.

**A finding on the way: orthonormality pins the ρ-sign convention.**
`RootDatum.rho_sign_exp`'s default is the coordinate staircase, whose own
docstring calls it "coordinate-convention-dependent" and validates it only
against U(N); `su_n` already overrides it.  On the hand-built B2 datum the
default gives `⟨L,L⟩ = 𝖖⁴` — orthonormality **fails** — while the SU(N)-style
always-even convention (sign `+1`) gives `1 − 𝖖² + 𝖖⁴`.  So the ρ-sign for a
fresh datum is **fitted by demanding Goal 2.1**, not derived.  That is a
legitimate axiom-driven choice and is labelled as such in the code; a
first-principles ρ-sign for a general datum is an open item.

## 4d. Rank 3 (user, 2026-07-27)

Root systems come from the repo's own surface, not from hand-written Cartan
data (user: *"The repo has root system tools" / "Do not reinvent the wheel"*).
`threed/root_data.py::GaugeDatum` is the validated root datum on the **flux
lattice** — roots are functionals `a(m) = Σ a_i m_i`, coroots are in flux
coordinates, `a(a^∨) = 2` enforced at construction — which is exactly
`RootDatum`'s convention, so `datum_from_gauge()` is a straight transcription
and the `su` / `so` / `sp` / `u` / `product` factories come along.  Validated:
it reproduces the spine's `su_n(N)` positive roots, Weyl order and Weyl vector
exactly at `N = 3, 4`.  (The *other* root-system surface is
`experiments/pure_g_cs_index.py`, carrying the full A–G classification incl.
E/F/G in **ambient** coordinates for the CS-index gates — not a flux-lattice
datum, so not what this solver consumes, but that is where exceptional data
lives if the solver is pushed past G₂.)

**SU(4) = A₃ — the validation, since it is the only rank-3 case with independent
ground truth (`PureSUNKAlgebra(4)`):**

| `m` | support | reps | unknowns | equations | Bareiss | `⟨L,L⟩(K=4)` | vs truth |
|---|---|---|---|---|---|---|---|
| (1,1,1) — adjoint | 13 | 1 | 37 | 2700 | unique | `1 − 2𝖖² + 2𝖖⁴` | **MATCH** |
| (1,2,1) | 19 | 2 | 38 | 4260 | unique | `1 − 𝖖²` | **MATCH** (13/13 sectors) |

All 2700 / 4260 equations verified against the solution.

**SO(7) = B₃** (no ground truth; self-check + orthonormality) — **all three
labels clean**:

| `m` | support | unknowns | equations | Bareiss | `⟨L,L⟩(K=4)` |
|---|---|---|---|---|---|
| (1,−1,0) | 13 | 13 | 1524 | unique | `1 − 𝖖² + 𝖖⁴` |
| (0,1,−1) | 13 | 13 | 1524 | unique | `1 − 𝖖² + 𝖖⁴` |
| (1,0,−1) | 13 | 13 | 1524 | unique | `1 − 𝖖² + 𝖖⁴` |

bar ✓ and (★) ✓ throughout; all 1524 equations verified in each case.
**Sp(6) = C₃** was still running when this was written (its `m = (1,−1,0)` has
support 19, the heaviest case attempted) — the numbers above are the ones
actually observed, and nothing is claimed for C₃.

**A free cross-datum check, set up but not yet run: D₃ ≅ A₃.**  Measured at
datum level: `GaugeDatum.so(6)` and `GaugeDatum.su(4)` have the same abstract
Cartan type up to relabelling of simple roots.  The subtlety is the global form
— `so(6)` is the SO form (flux lattice = coweight lattice) while `su(4)` is
simply connected, so they differ by a `Z₂`.  The meaningful comparison is SO(6)
restricted to the **coroot sublattice** = Spin(6) = SU(4), which is exactly what
`tropical_support` produces when seeded from a coroot-lattice `m`.  This is the
only cross-datum consistency test available at rank 3 that needs no new ground
truth.

**Cost note.**  At rank 3 the *solve* is cheap (the Weyl-covariance reduction
keeps unknowns in the tens); the wall-clock is dominated by the post-solve
self-check — `criterion_wrq` over all walls × support pairs, and the WRQ trace
with `|W| = 48`.  Roughly 280 s per B₃ label, against ~1 s of Bareiss.

## 4e. ⚠ PROPERNESS IS NOT VALID FOR GENERAL `e` (user, 2026-07-27)

**User report** (complaints collected from other sessions): *"properness is not a
valid constraint for general e, we should use the requirement that `L_{m,e}` acts
nicely on the space of symmetric functions, e.g. wavefunctions of
`L_{0,e'}|N]`, the Neumann boundary lines, 'B₂ at e ≠ 0 is unreachable' probably
because of properness."*

**Confirmed by measurement.**  Properness — "the numerator's degrees lie strictly
inside the denominator's span", used above as the executable form of the
`O(𝖖)`/KL condition — **fails on the true canonicals** from `(m,e) = (2,2)`
onward at SU(2):

| label | sector | `\|K_f\|` | window | TRUE degrees | proper? |
|---|---|---|---|---|---|
| (2,1) | ±1 | 2 | [1,3] | [1,3] | ✓ (exactly at the boundary) |
| **(2,2)** | ±1 | 2 | [1,3] | **[0,2]** / **[2,4]** | ✗ |
| (2,3) | ±1 | 2 | [1,3] | [−1,1] / [3,5] | ✗ |
| (2,6) | ±1 | 2 | [1,3] | [−2,0] / [4,6] | ✗ |

So properness would have **excluded the truth**.  It happens to hold exactly
where it was tested: `m = 1` at every `e` (a single interior sector, which stays
in `[1,3]`), the *centre* sector at any `e`, and everything at `e ≤ 1`.

**Scope of the damage.**  Every result recorded above is `e ≤ 1` at rank 1 and
`e = 0` at rank ≥ 2, i.e. inside properness's valid region — and the SU(3)/SU(4)
cases were verified against independent ground truth regardless of it.  So the
recorded numbers stand.  What does **not** stand is the *method* as implemented:
it cannot reach `e ≥ 2` at `m ≥ 2`, which is exactly the reported
"B₂ at `e ≠ 0` is unreachable".

**~~The open design question~~ — RESOLVED, see §4f.**  I had proposed a KL
recursion over already-built lower canonicals as the replacement pinning device.
**User ruling (2026-07-27): *"I do not see why you would need a KL recursion"*
and *"the condition that bubbling is `O(𝖖)` should be powerful enough to subsume
properness"* — and *"definitely impose the actual condition that bubbling is
`O(𝖖)` (a condition on `a_m(v)`, remember…) not proxies for it".***  Both points
are right and are now measured facts; §4f records the corrected device.  For the
record, the recursion was unnecessary for an elementary reason: adding
`c(𝖖)·L_{m',e'}` needs `c` bar-invariant *and* contributes at order `𝖖⁰` to a
cell required to be `O(𝖖)`, so `c ∈ 𝖖Z[𝖖]` and `c(𝖖) = c(𝖖⁻¹)` force `c = 0`.
And the true role of properness was never *selection* but *finiteness* of the
ansatz — a distinction I had blurred by calling it "the executable form of the
`O(𝖖)`/KL condition".

## 4f. THE CORRECTED DEVICE: (★) + bar + `O(𝖖)`, no properness (2026-07-27)

Implemented in **`experiments/oq_bubbling_solver.py` [retired 2026-09-19 with the type-A keystone; the measurement stands as recorded]**, which **supersedes**
`experiments/general_g_joint_bubbling_solver.py` (and the rank-1
`su2_joint_bubbling_solver.py`).

**Bar, made exact (measured, 110/110 cells at SU(2)/SU(3)/SU(4)).**  In the
f-presentation bar is plain `𝖖 ↦ 𝖖⁻¹` with `v` fixed, so its content depends on
whether the denominator is bar-stable.  It is: **`K_f(n)` is `k ↦ −k` symmetric
in every cell measured** — 0 violations — hence `D̄ = D` by relabelling, and

> bar-invariance ⟺ **each numerator coefficient `c_λ(𝖖)` is `𝖖`-palindromic**

with no `v`-flip and no shift.  This is what turns each unknown from a free
rational function of `𝖖` into an **integer vector**,
`c_λ = a_0 + Σ_{t≥1} a_t(𝖖^t + 𝖖^{−t})`, and makes the whole system linear over
`Z`.

**`O(𝖖)`, imposed literally on `a_n(v)`.**  Expanding `1/D` in the positive-`𝖖`
direction, a factor with `k > 0` contributes valuation `0` and one with `k < 0`
contributes `|k|`, so `val_𝖖(1/D) = S(n) := Σ_{(α,k) ∈ K_f(n), k>0} k`.  The
condition imposed is the actual one — every coefficient of `𝖖^g`, `g ≤ 0`, in
the series for `f_n` vanishes.  (Its per-coefficient *consequence*
`max_λ deg_𝖖 c_λ ≤ S(n) − 1` is used only as an audit statistic: it holds in all
110 cells and is **tight in 37** of them.)

**`O(𝖖)` is indispensable — measured, by accident.**  A truncation bug briefly
switched the `O(𝖖)` block off.  With (★) + bar only, SU(2) `(m,e) = (2,3)` still
solved **uniquely — to the wrong element**, which passes `W1` *and* (★):

    got   n=−1:  (𝖖⁻⁴+𝖖⁻²+1+𝖖²+𝖖⁴)v − v³        [deg 4 > S−1 = 3]
    want  n=−1:  v⁻¹ + (𝖖⁻²+1+𝖖²)v

So (★) + bar alone do **not** pin the canonical, and what kills the impostor is
precisely `O(𝖖)`.  This is the sharpest evidence in this note for the user's
claim that `O(𝖖)` is the load-bearing condition.

**The `v`-window: an anchored box, and only a cutoff.**  Properness anchored the
numerator's Newton polytope at `0`; the truth is anchored at the **Wilson
weights**.  Measured on the true SU(2) canonicals: the numerator of cell `n`
always lies in

    box(n)  =  wt(χ_e)  ⊕  Σ_α [0, mult_α(K_f(n))·α]        (dilated by `pad`)

— never below `−e`, never above `e + 2|K_f(n)|`.  `wt(χ_e)` comes from the
repo's own `wrq_torus.wilson`.  At `e = 0` it is the single weight `0` and the
box degenerates to the zonotope, **which is why every earlier `e = 0` result is
unaffected**.  Crucially the box does no selecting: at SU(2), `pad = 1, 2, 3`
return the *same* unique answer = the truth on
`(m,e) ∈ {(1,0),(1,1),(2,0),(2,1),(2,2),(2,3),(2,6),(3,0)}` — over-widening never
underdetermines.  The driver therefore ships a **box certificate**: every solve
is repeated one pad wider and the two elements must agree.

**Recovered at `e ≥ 2`, where properness failed outright.**  `(2,2)`, `(2,3)`
and `(2,6)` — inconsistent or wrong under properness — all now return the truth
with `W1`, (★), orthonormality `⟨L,L⟩ = 1 + O(𝖖²)`, and the box certificate.

**Measured with the corrected device** (all: unique, every equation verified,
`W1` ✓, (★) ✓, orthonormal ✓, box certificate at `pad+1` ✓):

| datum | `m` | `e` | vs ground truth |
|---|---|---|---|
| SU(2) | 1 | 0, 1 | MATCH |
| SU(2) | 2 | 0, **2**, **3**, **6** | MATCH (properness: wrong or inconsistent) |
| SU(3) | `(−1,−1)` | `(0,0)` | MATCH |
| SU(3) | `(−1,−1)` | `(1,0)` = **3** | MATCH |
| SU(3) | `(−1,−1)` | `(0,1)` = **3̄** | MATCH |
| SU(3) | `(−1,−1)` | `(1,1)` = **adjoint** | MATCH |
| **B₂ = SO(5)** | `(1,0)` | `(1,0)`, `(0,1)` | — (no ground truth; self-certified) |
| **B₂ = SO(5)** | `(0,1)` | `(1,0)` | — (9020 equations verified) |
| **B₂ = SO(5)** | `(1,1)` | `(1,0)` | — |
| **G₂** | `(1,0)` | `(1,0)` ≠ 0 | — (69 900 equations verified; `1 − 𝖖² + 2𝖖⁴`) |
| SU(4) (rank 3) | `(1,1,1)` | `(0,0,0)` | MATCH (113 938 equations verified) |
| **SU(4) (rank 3 DRESSED)** | `(1,1,1)` | **4**, **4̄**, **Λ² = 6** | **MATCH** — rank 3 *and* `e ≠ 0`, the region properness locked out |
| Sp(6) = C₃ (rank 3) | `(1,-1,0)` | `(0,0,0)` | — (806 220 equations verified; see §4h) |



**"B₂ at `e ≠ 0` is unreachable" — RESOLVED, and the reported diagnosis was
right.**  All four B₂ labels above solve uniquely with the corrected device,
each with every equation verified, `W1` ✓, (★) ✓, orthonormality
`1 − 3𝖖² + 6𝖖⁴` / `1 − 2𝖖² + 3𝖖⁴` / `1 − 𝖖² + 3𝖖⁴` ✓, and the `pad+1` box
certificate ✓.  Properness was exactly the blocker: B₂ has **no minuscule
cocharacter in the coroot lattice**, so every monopole bubbles, and dressing it
(`e ≠ 0`) moves the numerator off the zonotope that properness anchored at `0`.
(Caveat unchanged from §4c: the B₂/G₂ ρ-sign is **fitted** by demanding Goal 2.1
orthonormality, not derived — see the `rho_sign_exp` note there.)

*Coordinate warning (cost me one false alarm).*  `root_datum.su_n(N)` carries
cocharacters in the **coroot** basis and weights in the **fundamental-weight**
basis, while `PureSUNKAlgebra` labels are **ambient**.  For SU(3):
`m_datum = (−1,−1) ↔ m_amb = (−1,0,1)`, and `e_datum = (1,0), (0,1), (1,1)
↔ e_amb = (1,0,0), (0,0,−1), (1,0,−1)`.  Comparing `e_datum = (1,0)` against
`e_amb = (1,0,−1)` compares the **3**-dressed canonical with the adjoint-dressed
one and reports a spurious MISMATCH.

## 4g. Weyl covariance is SUBSUMED, not an axiom (user conjecture, confirmed)

**User (2026-07-27):** *"Weyl covariance in appropriate sense of
`Σ_m a_m(𝖖^m) U_m` should be a condition, but probably subsumed in the existence
of a good action on symmetric functions."*  **Measured: it is subsumed.**

`oq_bubbling_solver.joint_solve(..., weyl_cov=False)` drops the assumption
entirely — every interior cell gets its own independent unknowns instead of
being transported from an orbit representative, roughly doubling to
sextupling the unknown count (SU(2) `(2,3)`: 88 → 176; SU(3) adjoint:
120 → 728).  In every case the system is **still uniquely solvable**, the
solution is **automatically Weyl-covariant** (`f_{w(n)} = w·f_n` checked for
every `w` and every cell), and it **matches the truth**:

| case | unknowns (free) | unique | covariant | vs truth |
|---|---|---|---|---|
| SU(2) (1,0) | 20 | ✓ | ✓ | MATCH |
| SU(2) (2,0) | 116 | ✓ | ✓ | MATCH |
| SU(2) (2,2) | 156 | ✓ | ✓ | MATCH |
| SU(2) (2,3) | 176 | ✓ | ✓ | MATCH |
| SU(3) `e=0` | 488 | ✓ | ✓ | MATCH |
| SU(3) adjoint | 728 | ✓ | ✓ | MATCH |

So the selecting conditions are **three**, not four: (★), bar, `O(𝖖)`.  Weyl
covariance is a *theorem* of those on the cases measured, and the structural
imposition survives only as an accelerator (`weyl_cov=True`, the default).

*(The rank-3 rows run with the box certificate switched off — a `pad+1`
re-solve is thousands of unknowns there — and are covered by ground truth
instead, which is the stronger check.  G₂'s certificate was deferred for the
same reason and is now **done**: with numpy available, `pad = 2` re-solves and
agrees.)*

## 4h. What C₃ = Sp(6) exposed — two real bugs, both invisible in type A

Sp(6) was the first datum to fail, reporting a *seed inconsistency* at the wall
`α = (0,2,0)`.  It was not a failure of the axioms; it was two defects in my
implementation, and both were only reachable outside type A.

1. **The `𝖖`-window was guessed, not derived.**  An unknown `a_{rep,j,t}` reaches
   `𝖖`-degree `dg ± t`.  If a seed constant sits at a degree no unknown can
   reach, the row has no unknown columns and a nonzero constant — which *looks*
   inconsistent while the window is merely too narrow.  `T` is now computed from
   the assembled equations (never below the measured `S(n) − 1` bound).

2. **The box stepped in units of `α`, not of the primitive `α₀`.**  C₃ has long
   roots `2ε`, so along those the box skipped every lattice point strictly
   between the segment's endpoints — exactly the monomials a bubbling cell needs
   to cancel the seed residue on that wall.  In type A every root is primitive,
   so this could never show up.  **B₂'s `(2,−2)` and `(0,2)` are divisible too**,
   so the B₂ runs recorded above were coarser than intended; they were re-run
   with the finer box and reproduce (driver: 18/18 MATCH, all certificates).

A third, performance-only defect: `pad` was dilating *each factor's range*, which
at 18 factors inflates the box by the sum of all 18 roots.  It now dilates the
polytope once per unit.

**Why this matters beyond the bug.**  The finer box is strictly *larger*, and
every previously-solved case returns the same unique answer inside it — one more
measurement that the box bounds without selecting.

## 4i. The ρ-sign is DERIVED, not fitted (user, 2026-07-27)

The device carried one fitted constant.  `_TRIVIAL_RHO_SIGN` — the ρ-twist sign
exponent `≡ 0` used for B₂, G₂ and everything built through `datum_from_gauge` —
was **pinned by demanding orthonormality**, and its own docstring said so: *"a
first-principles derivation of the ρ-sign for a general datum is an open item."*
The user's reading was right — *"I think the other session got ρ with
orthonormality, but there should be a more principled way"* — and there is:

    rho_sign_exp(k)  =  Σ_{α>0} ⟨α, k⟩  =  ⟨Σ⁺, k⟩,        Σ⁺ = Σ_{α>0} α

the datum's own Weyl vector (`RootDatum.weyl_vector`, doubled) paired with the
cocharacter.  Coordinate-free, hence transportable to any datum.  Implemented as
`general_g_joint_bubbling_solver.weyl_vector_rho_sign`.

**It is a CHARACTER OF `π₁(G)`, and that is the whole content.**  Measured on
every datum the repo carries: `k ↦ ⟨Σ⁺,k⟩ mod 2` is additive, and it kills
every simple coroot (`⟨Σ⁺, α_i^∨⟩ = 2`), so it vanishes on `Q^∨` and
descends to

    X_*(T)/Q^∨  =  π₁(G)  ⟶  {±1}.

The sign is the image of the monopole's GNO class in `π₁(G)`.  It is nontrivial
exactly on data that are not simply connected *and* whose `ρ_weyl` is not a
weight — measured trivial for SU(N), Spin(5), Spin(7), Sp(n) (`π₁ = 1`), trivial
for U(3) (`π₁ = Z` but `Σ⁺ = (2,0,−2)` is even), nontrivial for SO(5), SO(7)
(`π₁ = Z/2`) and for U(2), U(4).  This dovetails with #1040's ruling that **the
global form is part of the datum**: the sign is precisely the datum-level
invariant that global form controls.

> **CORRECTION to the first write-up of this section.**  It claimed the sign was
> "the missing half of `atom_phase`" — that the phase `−⟨ρ_weyl,m⟩` is a
> half-integer exactly when the sign is `−1`, so one quantity split in two.
> **U(2) refutes it**: the sign is nontrivial at `m = (1,0)` while
> `u_n(2).atom_phase((1,0)) = 0`, perfectly integral.  The correct statement is
> that both are governed by the SAME character above, not that they are two
> halves of one number.  `atom_phase`'s default IS half-integral exactly when
> the character is nontrivial, but that is repairable by the D10 freedom to add
> a **linear** (coboundary) term — and such a term exists iff the group has a
> central torus.  Measured: the space of Weyl-invariant linear functionals is
> **1**-dimensional for U(2)/U(3) (the direction `Σ_j m_j`, which is exactly
> what `u_n`'s historical `(N−1)/2·Σ_j m_j` shift uses) and **0**-dimensional
> for SU(3), SO(5), SO(7), Sp(3).  So U(N) repairs the parity with a coboundary
> and a semisimple datum cannot.

**Measured** (`experiments/` probe, 17 data, every cocharacter with `|k_i| ≤ 2`):

| datum | agreement with the existing hook |
|---|---|
| `u_n(2)`, `u_n(3)`, `u_n(4)` | **exact** (25/25, 125/125, 625/625) |
| `su_n(2..5)` | exact (reproduces the documented "rank-≥2 fix") |
| `b2_datum`, `g2_datum` | exact — **the fitted `+1` is derived** |
| `GaugeDatum` su(3), su(4), sp(2), sp(3), u(3) | exact |
| `GaugeDatum` **u(2), so(5), so(7)** | **differs** — see below |

That the U(N) case is *exact* is not luck: in the `e`-basis the `i`-th coordinate
of `Σ⁺` **is** `N−1−2i`, so the "coordinate staircase" is this same formula
written in one basis — which is precisely the coordinate-dependence
`rho_sign_exp`'s own docstring warns about.

**Why the B₂/G₂ fit worked.**  `⟨Σ⁺, α_i^∨⟩ = 2` for every simple coroot, so on
the **coroot lattice** `⟨Σ⁺, k⟩` is always even and the sign is *forced* to `+1`.
`b2_datum`/`g2_datum` carry cocharacters in the coroot basis, so `+1` was never a
free choice.  The fit was recovering a theorem — which is also why the coordinate
staircase *failed* there (`⟨L,L⟩ = 𝖖⁴`): it is the right formula in the wrong
coordinates.

**And it corrects a real defect.**  `+1` is right only when the cocharacter
lattice *is* the coroot lattice.  Where the flux lattice is strictly bigger,
`⟨Σ⁺, k⟩` is odd on half of it.  The decisive case is **U(2) reached two ways**:
`GaugeDatum.u(2)` under the constant disagrees with the repo's certified
`u_n(2)`, while the derived law agrees with it exactly.  So the constant was
measurably wrong for U(2), and by the same mechanism for SO(5)/SO(7) in the flux
presentation.  Nothing already published moves: the `GaugeDatum` data actually
solved (SU(3), SU(4), Sp(6)) are all-even, and B₂/G₂ used the hand-built
coroot-basis data — driver re-run after the change, all cases recovered, all
certificates, 22/22 elements bit-identical.

**~~Open, and needing the user.~~  BOTH QUESTIONS NOW SETTLED — see Plan 24 D31
(2026-07-29).**  `RootDatum.atom_phase` no longer raises at odd `⟨Σ⁺,m⟩`; it is
total, and SO(7) *is* reachable — it builds and certifies.  The half-integer never
had to be carried anywhere, because the algebra needs the phase only through its
coboundary `δS̃`, an integer even where `S̃` is not.  The original two questions,
with their answers:

1. ~~Should `RootDatum.rho_sign_exp`'s **default** become `⟨Σ⁺,k⟩`?~~  **YES, and
   it did** (D31): the law is `rho_sign_exp(k) = atom_phase_doubled(k) + ⟨Σ⁺,k⟩`,
   which subsumes the staircase *and* all three overrides — `su_n`'s explicit
   override is retired, 0 mismatches over 1410 cocharacters, and it also retires
   `repo_audit` A29 (the staircase disagreed with `product_datum` on `U(1)²` at
   12 of 25 cocharacters; this law agrees everywhere, being additive over blocks).
2. ~~Whether `atom_phase` should carry a `𝖖^{1/2}` where the half-integer is
   intrinsic.~~ **RESOLVED, and the answer is NO** (user ruling, 2026-07-27:
   *"square roots of `𝖖` are too dangerous physically, you are not allowed to use
   them"*).  The canonical basis is over `Z[𝖖^±]` and that is the whole scalar
   ring.  This is not only a prohibition — **the route was already measured dead
   in this repo**, and it was proposed here without checking, which is exactly
   the failure the false-friends discipline exists to prevent.
   `experiments/so3_wrq_spinorial_obstruction.py` (re-run 2026-07-27) shows, for
   the odd-root-height spinorial line `H_0 = ω^∨` of SO(3).  ⚠ **Those numbers all
   reproduce, but their CONCLUSION is retracted (D31)** — they measure
   `_psi_monomial_data` materialising `M(m) ∝ (−𝖖)^{⟨ρ,m⟩}`, the square root of the
   measure, which the tier never needs.  The shipped path now gives
   `I(H₀,H₀) = 1 − 𝖖² + 𝖖⁴ + 𝖖⁶ − 𝖖⁸` == the BPS oracle, with **no `𝖖^{1/2}` and no
   `i`**.  The ruling below stands and was never the blocker.  Historical record:

   * every integer atom-phase fails — `I(H₀,H₀)[𝖖⁰]` is `0` or `−1`, never `1`,
     and `H₀²` is never bar-invariant (`S = 0, −m, −⌊m/2⌋, −⌈m/2⌉, −m²`)
     — *measured on override data, which moves ρ too and, since D31,
     double-corrects against `cocycle_R`*;
   * *simulating* `𝖖^{1/2}` does **not** rescue it either:
     `I_WRQ(H₀,H₀) = −𝖖^{−1}·I_BPS(H₀,H₀)` exactly (through `𝖖¹²`), and the kernel
     carries an `i` — *both are that one materialised monomial, not the algebra*;
   * the **standard-`Z²` BPS chart** carries it cleanly over `Z[𝖖^±]`:
     `I(H₀,H₀) = 1 − 𝖖² + 𝖖⁴ + 𝖖⁶ + O(𝖖⁷)`, `H₀² = L_{1,0}`
     (`implementations/pure_so3.py`; SO(3) = SU(2) with `(M,E) = (2m, e/2)`).

   ⚠ **The conclusion drawn here is RETRACTED (Plan 24 D31, 2026-07-29).**  It read:
   *"So an odd-`⟨Σ⁺,m⟩` cocharacter is not an atom on this torus at all; the
   coweight-atom frame is the wrong frame for it, and `atom_phase`'s raise is the
   CORRECT behaviour"* — with the open question being *"what is the right frame for
   the odd sectors of the SO-form `B_n`"*.  **Every measurement above reproduces; the
   inference does not.**  The torus never needs the phase as a *number*, only through
   its coboundary `δS̃`, which is an integer even where `S̃` is a half-integer, because
   `π = ⟨Σ⁺,·⟩ mod 2` is Weyl-invariant *and* additive.  `wrq_torus.cocycle_R`
   restores the honest phase through it, and odd-`⟨Σ⁺,m⟩` charges are ordinary atoms
   of this torus: SO(3)'s `H₀` builds bar-invariantly with `H₀² = L_{(2,0)}` and
   `I(H₀,H₀)` equal to the `PureSO3KAlgebra` oracle, SO(5)/SO(7) build and are
   orthonormal.  The `−𝖖^{−1}` and the `i` recorded above are both that one
   materialised monomial — `_psi_monomial_data`'s square root of the measure — and
   neither is a `𝖖^{1/2}` the algebra needs.  The standard-`Z²` BPS chart remains a
   correct independent presentation (it is the oracle D31 was certified against), but
   it is no longer the *only* frame that carries these lines.

## 5. Implications, and one contract question

> **✅ ITEM 1 RESOLVED (user ruling, 2026-07-27): (★) IS the guard, and a
> (★)-guarded solve is licensed.**  The reason given is an axiom, not a
> plausibility argument (user): *"the solver uses known axioms together with the
> assumption that Neumann lines are generated by `L_{0,e'}|N]` and thus the
> differential operators associated to `L_{m,e}` act on the space of symmetric
> functions."*  That is exactly the module-side equivalence derived in §2, so the
> "what would settle it" below is answered by the axiom rather than by a
> counterexample search.  **Landed:** `star_bubbling.py` (the promoted, guarded
> solver — unique + verified exactly over `Z` + W1 + (★) + pad+1 box certificate)
> and `implementations/pure_g_abe_kalgebra.py::PureGAbeKAlgebra` (one general-`G` class
> over a `RootDatum`, routes guarded individually and falling through on
> failure).  The `AbeKAlgebra` tier still exposes **no** solve entry point.
> Certified: at `su_2()` the solved chart `==` `PureSU2KAlgebra`'s fully-native
> build — so **the adjoint-monopole fiber that Plan 24 called "the single
> irreducible input" is now derived, not supplied**, with the generic
> product-and-peel taking over at `m ≥ 2` — and at `su_n(3)` `==`
> `PureSUNKAlgebra` (3/3).  Rulings + the session's findings (global forms, the
> odd-`⟨Σ⁺,m⟩` (since D31: builds, no honest-fail), the G₂ `e = 0` regression, the singular-`e` box
> anchor): `restructuring_plans/24_nonminuscule_groups/decisions.md` D5/D6.

1. **The missing post-hoc guard (HUMAN-DECIDE).**  CLAUDE.md bars a *solve* on
   the abelianized-generator tier because "there is no post-hoc guard: a solve
   can land outside the span, and orthonormality and the M-test are pairings
   defined *within* the span, so both pass on off-span garbage."  Criterion (★)
   is **not** a pairing within the span — it is a regularity condition on the
   presentation, and the negative controls show it rejects off-span
   perturbations that the self-norm cannot see.  If (★) is a genuine membership
   test, it is exactly the guard whose absence motivates the rule, and a
   guarded solve on this tier would become legitimate.  **This is a
   contract-surface question and is not being acted on.**  What would settle it:
   a proof (or a determined search for a counterexample) that (★) + W1/W2
   characterises the canonical basis.

   **Partial evidence, measured 2026-07-27** (`experiments/star_rejects_missing_bubbling.py` [retired 2026-09-19 with the type-A keystone; the measurement stands as recorded]).
   The `pure_un_joint_fiber_defect.md` offender's mechanism is stated there
   exactly: `_cf_joint_fiber` works at FIXED magnetic charge and *"has no
   mechanism to reach the lower-magnetic bubbling correction"*, so its output is
   **missing bubbling** — and (★) is precisely a condition coupling cells across
   magnetic charge.  Tested on the mechanism: strip the bubbling from a genuine
   canonical (keep the leading Weyl orbit only) and ask whether (★) notices.

   | | SU(2) (15 labels) | SU(3) (8 labels) |
   |---|---|---|
   | (★) on the TRUE canonical | 15/15 pass | 8/8 pass |
   | (★) on the STRIPPED seed | **0/15 pass — all rejected** | **0/8 pass — all rejected** |
   | label-map sanity (seed ≡ truth on extremal cells) | 15/15 | 8/8 |

   So (★) detects the defect's failure mode with no exceptions in 23 labels,
   whereas the bar-blind `𝖖⁰` self-norm did not.  **Caveat:** this tests the
   *mechanism*, not the literal `t₂`/`t₃` elements — those would need the retired
   `_cf_joint_fiber` solve resurrected — and it is evidence, not the proof the
   HUMAN-DECIDE asks for.  (The sanity row is not decoration: it caught a
   negated coroot coordinate in the first version of this probe, which had made
   the SU(3) column meaningless.)

2. **Groups with unknown bubbling (Plan 24).**  Plan 24's open items are exactly
   "closed-form SU(N) bubbling (Kostka–Foulkes)" and "beyond type A".  R3 + R4
   give a route that never needs the bubbling formula: the extremal orbit is the
   closed-form Levi character (always available), R3 fixes the denominators, (★)
   fixes the polar parts, W1/W2 picks the representative.  R7 shows (★) is
   already group-general (affine Weyl, no type-A input) and *demonstrates the
   discrimination on a real defect* — it rejects `build_canonical`'s
   bubbling-free SU(2) seed.  What remains is to run the R4 **reconstruction**
   (not just the test) on the general-G substrate and check the recovered
   bubbling against `PureSU2KAlgebra` / `PureSUNKAlgebra`; if that closes, the
   route is executable at any datum.

3. **`N(k)` — settled, and it pays.**  See R6: the criterion is not merely
   *satisfied* by the CS-twisted state, it **derives** the state's quadratic
   Gaussian `𝖖^{kΣa_i²}` (and its sign) from the wall mismatch, with the
   linear `tr(a)` left free as the framing convention.  This is the first
   instance of the criterion being used as a *derivation tool* rather than a
   test, and it is the template for item 2: an unknown ingredient (there the CS
   coupling, here the bubbling) is pinned by demanding (★).

## 5b. The spanning test, formulated (2026-08-25)

**Why this is the question.**  The one `KAlgebra` axiom that does *not* follow
formally from the `AbeKAlgebra` axioms is **closure** — that `L_a·L_b` lies in
the `Z[𝖖^{±1}]`-span of the canonicals (`kalgebra.md`, "The `KAlgebra` axioms
from the `AbeKAlgebra` axioms").  Since (★) makes each `L_{m,e}` a
`𝖖`-difference operator preserving `Λ = R(G)`, and such operators are closed
under composition, one containment is free:

    span_{Z[𝖖^{±1}]}{L_{m,e}}  ⊆  𝒜(★) := { x : x·Λ ⊆ Λ }.

So the algebra axiom **is** the converse containment — in the user's words
(2026-08-25), *"the `L`'s span the (★)-algebra"*.  That is also §5 item 1 of this
document in its other guise ("a proof … that (★) + W1/W2 characterises the
canonical basis").  What follows is the shape of the *determined search* that
item asks for.  It is a formulation, not a result: nothing here has been run.

### The test

Both sides are `Z[𝖖^{±1}]`-modules, so compare them where both are finite: fix a
truncation and compute **two ranks of explicit integer matrices**.

1. **Choose the truncation from the LABELS, not from the cells.**  Pick a finite
   set of labels `B`; let the cells be the union of their tropical supports and
   the coefficients range over the union of their `box_points` boxes, dilated by
   `pad` — exactly the ansatz the solver already builds.  Choosing the box first
   and then asking which labels fit would truncate canonicals mid-element and
   make the comparison meaningless.
2. **Left side.**  Unknowns are the numerator coefficients `c_{n,λ,k}` (cell `n`,
   weight `λ` in the box, `𝖖`-degree `k` in the window).  Impose **(★) only** —
   no seed, no W2, no `O(𝖖)`.  These are the same `sym_rows`/`expand` rows
   `joint_solve` already generates.  Then
   `d★ = nvars − rank(system)`, the dimension of the (★)-solution space.
3. **Right side.**  Each `𝖖^k·L_{m,e}` that fits entirely inside the truncation is
   an explicit vector in the *same* coefficient space.  `dL = rank` of the matrix
   of those vectors.
4. **Verdict.**  Spanning holds in this truncation iff `d★ == dL`.

### What each outcome means

| | reading |
|---|---|
| `d★ == dL`, stable under `pad → pad+1` | spanning holds there; repeated across `(G, N)` this is the determined search §5 asks for |
| `d★ > dL`, stable under `pad` | **a counterexample** — an explicit (★)-element outside the span.  Extract a solution vector not in the span, re-verify (★) on it exactly, and check `decompose` refuses it |
| `d★ < dL` | impossible — canonicals satisfy (★).  A harness bug, and the built-in consistency check |

### The three ways this test can lie, and the controls for each

* **Truncation artifacts.**  An element supported partly outside the truncation
  is invisible, so a bare equality proves nothing on its own.  Require the
  *verdict* to be stable under `pad → pad+1` — the same box-certificate
  discipline `solve_canonical` already applies ("the box is a cutoff, never a
  selector").  A rank equality that moves with `pad` is a window artifact.
* **Unlucky primes, in the direction that fakes a discovery.**  A rank computed
  mod `p` can only be **≤** the true rank, so `d★` computed mod `p` can only be
  **over**-estimated — which is exactly the direction that manufactures a
  spurious counterexample.  Compute at two primes and take the larger rank; and
  never report `d★ > dL` without verifying the extra solution exactly over `Z`.
* **The wrong `Λ`.**  (★) is regularity against the rep ring of **this form**,
  not of the cover (D24).  Positive control: at `m = 0` the answer is known —
  `𝒜(★)` in the magnetic-zero sector is multiplication by `R(G)`, with basis
  `{χ_e}` — so the test must return equality there.  Negative control: run a
  non-simply-connected form against the **cover's** `R(G̃)` and the two numbers
  must come out *different*, showing the test is sensitive to which `Λ` it is
  asked about.

### One sector where spanning is a THEOREM, not a measurement

At `m = 0` the statement is provable, and it is worth having before any code is
written — it turns the positive control above from "the answer is known" into
"the answer is proved", and it localizes where a failure could possibly be.

An element supported only at magnetic zero acts by multiplication by its residual
`f(v)`.  If it preserves `Λ = R(G)`, apply it to `1 ∈ R(G)`: then
`f = f·1 ∈ R(G)`.  Conversely `R(G)` is a ring, so every `f ∈ R(G)` preserves it.
Hence

    𝒜(★) ∩ {magnetic 0}  =  R(G)   exactly,

and the canonicals there are `L_{0,e} = χ_e`, a `Z[𝖖^{±1}]`-basis of it.  So
**spanning holds in the magnetic-zero sector**, and any failure of the spanning
statement must occur at **nonzero magnetic charge** — which is also where the
whole difficulty of this tier has always been.

Note this is the same degeneracy that makes the Wilson lines fall out of the
axioms with an empty system (`kalgebra.md`): at `m = 0` the tropical support is
one cell, so there is nothing to bubble and nothing to solve.

### Implemented and run (2026-08-25) — spanning HOLDS where tested, and the rank design is superseded

Battery: `experiments/spanning_test.py`.  Every claim below is gated on controls,
and the controls did the work: **three successive readings were wrong and were
caught**, which is the only reason the final one is trustworthy.

**Final status.**

* **`m = 0`** — spanning is a theorem (proved below), and the test reproduces it:
  `d★ == dL` at SU(2) (two label sets), SU(3), Spin(5).
* **Nonzero `m`** — at SU(2) `m ≤ 1`, **12 of 12** vectors of the (★)-nullspace
  decompose **exactly** into canonicals (`decompose` + reconstruction check).
  **Spanning holds there.  No counterexample.**

**The rank comparison of §5b is NOT a valid membership test, and this is the
lesson.**  It compares `dim` of the (★)-solution space against the dimension
spanned by canonicals that fit a truncation.  But a `Z[𝖖^{±1}]`-combination of
canonicals can lie **inside** a `𝖖`-window while its individual `𝖖`-shifts lie
outside — the out-of-window parts cancel.  So the canonical side is
systematically undercounted, and the "surplus" it reports is an artifact.  The
correct oracle is **`decompose` plus exact reconstruction** on each nullspace
vector: it answers membership directly and needs no window bookkeeping at all.

**The path to that conclusion, kept because each step is a usable lesson.**

1. First reading: a constant gap as the `𝖖`-window widens ⇒ "a boundary term, no
   counterexample".  An inference that *fits* the numbers, not a control.
2. Second: the direct edge test — with `E` the span of basis elements at the
   extreme `𝖖`-degrees, "the surplus is at the edge" is `N ⊆ C + E`, testable as
   `rank(C+E+N) == rank(C+E)`.  It **failed** every time, which retired reading 1
   and produced "a strong candidate counterexample".
3. Third, decisive: rebuild a surplus vector as an actual element and ask
   `decompose`.  It decomposes **exactly**, 12/12.  So reading 2's `C + E` was
   simply not the right notion of "explained by the window" — the cancellation
   mechanism above is, and no rank test on whole-fitting shifts can see it.

The `d → f` inversion that unblocked step 3 was thought impossible earlier in the
same session ("`TorusRational` has no division").  It is easy: `ψ_n` has a
**monomial** numerator over wall-factor denominators, so
`1/ψ_n = (∏ wall factors) × (inverse monomial)`, both already in the ring.
Controlled by round-tripping four canonicals — chart recovered exactly,
`criterion` ✓, `well_formed` returning the right label.

**Two corrections to the formulation that survive.**

1. **`criterion` is only half of (★).**  It tests residue cancellation; an
   operator must *also* be Weyl-equivariant to act on symmetric functions at all.
   Canonicals have that by construction, so the repo never had to test it — an
   ansatz with unknowns on every cell must.  At `m = 0` it is the entire
   condition (no walls), which is why that control caught it at once.
2. **Covariance is not a permutation of coefficients.**  `weyl_act`
   re-canonicalizes the denominator, so the image carries a compensating
   monomial; imposing covariance as coefficient identifications is right only
   where the denominator is trivial — i.e. only at `m = 0`.  The ansatz must be
   covariant **by construction**, which is `joint_solve`'s design and now has a
   stated reason.

### The right frame for this (user, 2026-08-25) — "lie in and generate"

*"In concrete examples (quivers etc) I think the expectation is indeed that the
`⋆` span **is** the BFN algebra.  Our job then is to argue that our `L_{m,e}` lie
in and generate the BFN algebra."*

That ordering dissolves the puzzle rather than explaining it away.  `𝒜(★)` is an
algebra for free; **lie in** is (★) for each canonical, which the tier enforces;
**generate** is the spanning statement.  Closure is then a corollary — a basis of
an algebra multiplies inside it — not a miracle.  The surprise was only ever an
artifact of taking closure as primitive and spanning as the thing to check.

**Where the identification's weight actually sits (user ruling, 2026-08-25).**
Asked whether "the `⋆` span is the BFN algebra" should be read as a definition or
a conjecture, the user: *"True, I am assuming a BFN result extends to K-theoretic
BFN."*  So the chain is explicit, and each link is a different kind of claim:

1. a **result** in the cohomological BFN setting (the source);
2. its **extension to K-theory — assumed, not proved**;
3. `𝒜(★) ≅` that K-theoretic algebra.

Link 2 is the one carrying the risk, and the repo's own record says why it cannot
be waved through: Plan 24 (user direction, 2026-06-10) has K-theoretic BFN *"not
really written"*, the working definition being the K-theory of the
Cautis–Williams category, and `AbeKAlgebra` ruling D2 has the literature's
abelianization as "related and **non-load-bearing**".  None of that makes the
assumption wrong — it makes it an assumption, which is how it is recorded here.

**"Generate", measured with the `decompose` oracle** — every vector of the
(★)-nullspace rebuilt as an element and decomposed, with the control that
`decompose` returns each canonical as itself:

| datum | basis | rows | nullspace dim | decomposes exactly |
|---|---|---|---|---|
| SU(2) `m ≤ 1` | 18 | 24 | 8 | **8** |
| SU(2) `m ≤ 2` | 147 | 328 | 39 | **39** |
| SU(2) `m ≤ 2` dressed | 299 | 692 | 73 | **73** |
| SU(3) `m ≤ (1,1)` | 78 | 286 | 10 | **10** |
| U(2) `m ≤ (1,0)` | 2 | 0 | 2 | **2** |
| Spin(5) `m ≤ (1,1)` | 45 | 128 | 8 | **8** |
| Sp(2) `m ≤ (1,1)` | 198 | 992 | 26 | **26** |
| SO(3) `m ≤ 3/2` | 84 | 132 | 36 | **36** |
| SO(3) `m ≤ 3/2` dressed | 108 | 152 | 50 | **50** |
| PSU(3) `ω_1^∨` | 4 | 0 | 4 | **4** |
| SO(5) `ω_1^∨` | 4 | 0 | 4 | **4** |
| Sp(2)/Z₂ `ω_2^∨` | 4 | 0 | 4 | **4** |
| G₂ `m ≤ (1,0)` | 640 | 6852 | 30 | **30** |
| **U(3) `m ≤ (1,0,0)`** | 2 | 0 | 2 | **2** |
| **SU(4) `m ≤ (1,0,0)`** | 425 | 2858 | 12 | **12** |
| **Sp(3) `m ≤ (1,0,0)`** | 78 | 404 | 10 | **10** |

**318 nullspace vectors over 16 truncations, every one decomposing exactly**,
`unverified 0` throughout.  **Rank 3 is in the table** — U(3) and SU(4) at
`m ≤ (1,0,0)` — and it is there only because of the solver work: the user's note
that *"times will dilate"* with rank was right, and D42's dedup plus D47's
shortest-row-first ordering contracted them by ~2.8×, putting SU(4) at ~8 minutes.
U(3) is minuscule at that charge, so its truncation is trivial (basis 2) and it
demonstrates the machinery rather than testing it; **SU(4) (basis 425, 2858 rows)
and Sp(3) — non-simply-laced — are the real rank-3 rows**.

**And the wall is still there, one charge further out.**  Two rank-3 attempts did
not make it, both recorded rather than dropped:

* **SU(4) `m ≤ (1,1,1)` is not a second charge at all.**  At SU(4) the cocharacters
  `(1,0,0)`, `(1,1,0)`, `(1,1,1)` and `(0,1,0)` **all fold to the same label**
  `(1,1,1)` — 13 cells — so the run reproduces the `(1,0,0)` row to the digit
  (425 / 2858 / 12).  Identical numbers are the tell, the same one that exposed
  the retracted non-simply-connected rows above; both batteries now flag a
  repeated `(basis, rows, vectors)` signature automatically.  The genuinely
  different next charge is `(2,0,0) → (2,2,2)`, at **55 cells** rather than 13.
* **Sp(3) `m ≤ (1,1,0)` (19 cells, 198 wall-slots) was abandoned after ~40
  minutes** without finishing.  The exact nullspace scales roughly cubically in
  the ansatz size — SU(4)'s basis 425 takes ~8 min — so a basis in the low
  thousands is out of reach at the current cost.  Not a failure of the machinery;
  the reach after D42 + D47 is *rank 3 at a small charge*, and one charge beyond
  that the wall returns.  **G₂** is the hardest datum the tier has —
non-simply-laced, no minuscule cocharacter, the richest wall structure (and by
far the slowest: ~20 min against seconds for the rest, which is the practical
ceiling on pushing this to higher rank).  The last six rows are the
non-simply-connected forms, reached through fractional `m`; how they got there,
and what the earlier version of this table got wrong about them, is next.

🛑 **RETRACTED, same day: the last three rows did NOT test a different
condition.**  The first version of this section claimed that PSU(3), SO(5) and
Sp(2)/Z₂ tested (★) against the *form's* module `Λ_H = R(G̃)^H`, "a strictly
different condition — the D24 distinction".  Wrong, in both halves of the form,
and the tell was sitting in the output:

| | basis | rows | decomposes |
|---|---|---|---|
| SU(3) `m ≤ (1,1)` vs **PSU(3)** | 78 = 78 | 286 = 286 | 10 = 10 |
| Spin(5) `m ≤ (1,1)` vs **SO(5)** | 45 = 45 | 128 = 128 | 8 = 8 |
| Sp(2) `m ≤ (1,1)` vs **Sp(2)/Z₂** | 198 = 198 | 992 = 992 | 26 = 26 |

Identical to the last digit in all three — the signature of a run carrying no
form data at all.  **The electric half cannot enter these rows, as a matter of
theorem, not of harness omission.**  The wall condition is `Res_n + Res_p = 0`;
that *is* the `χ_0 = 1` member of the `χ_e` family, and `χ_0` is the identity of
`Λ_H` for every form, so the extra characters of the cover impose nothing beyond
it.  `star_bubbling.preserves_neumann_module` already says this of the wall test
it shares with the harness — `criterion` is *"a wall-by-wall residue test on
`x.datum` alone, **blind to which electric charges the form actually has** … at a
non-simply-connected form (★) has to be asked of the module, not of the walls"* —
and the corollary is measurable: `wall_is_probed` returns `False` only for a
predicate admitting **nothing**, so 0 walls are dropped at PSU(3), SO(5),
Sp(2)/Z₂ and SO(3), and the `dropped_walls` branch of `joint_solve` is
unreachable for any real form.  Threading `admits_e` into the harness was tried,
measured inert, and removed rather than kept — a knob that cannot change the
answer reads to the next session as though the form were handled.

**The magnetic half is where a form-dependent run lives**, and the harness missed
it for the reason `global_form.adjoint_lines` already warns about in capitals:
the extra cocharacters are **fractional** in coroot coordinates, so an
integer-only sweep *"silently omits exactly the lines that make the form
interesting"*.  At `adjoint_lines(su_2())` the spinorial `ω^∨` is `m = 1/2`;
integer `m = 1` is the SU(2) adjoint monopole, not a new line.  A fractional `m`
changes the cells, and with them the basis, the walls and the canonicals — so
those runs are a different problem, and they are what the repaired `main()` now
runs.  The old measurements stand as measurements (three more simply connected
truncations where "generate" holds); the count of **non-simply-connected forms
tested was 3, and was 0**.

**Repaired, with `m ∈ P^∨ \ Q^∨`:**

| datum | basis | rows | nullspace vectors decomposing exactly |
|---|---|---|---|
| **SO(3)** `m ≤ 3/2` (spinorial + integer) | 84 | 132 | **36** |
| **SO(3)** `m ≤ 3/2` dressed (`e = 2`, the minimal admitted Wilson) | 108 | 152 | **50** |
| **PSU(3)** `ω_1^∨ = (2/3, 1/3)` | 4 | 0 | **4** |
| **SO(5)** `ω_1^∨ = (1, 1/2)` | 4 | 0 | **4** |
| **Sp(2)/Z₂** `ω_2^∨ = (1/2, 1/2)` | 4 | 0 | **4** |

`unverified 0` throughout.  **SO(3) `m ≤ 3/2` is the only stressing run of the
five**: 84 basis elements against SU(2) `m ≤ 2`'s 147, on a *different* cell set
(half-integer cells), with 132 non-trivial rows — not any simply connected run in
disguise.  The three rank-2 fractional coweights give `rows 0`: their truncations
carry no wall at all, so (★) is vacuous and the run demonstrates the machinery
rather than testing it.

That is **not** a harness defect, and it was predicted: `preserves_neumann_module`
records the same thing of `criterion` — *"measured to be nearly **vacuous** at a
fractional cocharacter — at `su_2` `m=(½,)` the entire wall set is `{((2,), 0)}`,
one wall at `k = 0`, so a bare leading orbit passes it while the bare orbit at
even `m=(1,)` correctly fails"*.  So the wall test thins out precisely where the
non-simply-connected form is interesting, which is the second reason (after the
`χ_0` argument above) that a wall-row spanning test is the wrong instrument for
D24.  Reaching those forms properly needs `m ≤ 3/2`-style truncations that
*combine* fractional and integer sectors — which is why the SO(3) rows have
content and the rank-2 ones do not.

### The module condition — implemented, and it adds nothing

The gap named above is now closed, with a different answer than expected.  (★)
asked **of the module** is `preserves_neumann_module`'s other half: `x·χ_μ` must
decompose onto **admitted** characters only, a `left-module` failure otherwise.
That is linear in `x` (`c_ν = [v^ν](P·Δ')`), so it becomes rows in the reduced
coordinates of the wall-nullspace — `module_rows` in the battery.

Note the direction: **(★)-for-the-form is strictly STRONGER than
(★)-for-the-cover**, not weaker.  The polynomiality half is shared (the `χ_0`
argument above), and this half only *adds* rows.  My earlier reasoning had it
backwards — a smaller module looked like a weaker requirement, but the image has
to land in the smaller module too.

**Two controls fire before any of it counts:**

* *Polynomiality (the `χ_0` argument, measured).*  Every `x·χ_μ` here must
  already be polynomial, since `Res(x·χ_μ) = Res(x)·χ_μ|_wall` and `x` satisfies
  the wall rows.  Measured `non-poly 0` on every run — the derivation is not just
  argued.
* *The negative control, which is the whole point.*  `module_rows` returns **zero
  rows on every truncation the battery runs**, and a function that always returns
  zero is indistinguishable from the no-op this section was corrected for.  So it
  is made to fire: build the ansatz on **SU(3)** with `χ_{ω_1}` among the labels —
  legal there — and ask the **PSU(3)** condition of that space.  `ω_1 ∉ Q`, so
  PSU(3) must cut it.  It does: **SU(3) 0 rows, dim 3 → 3; PSU(3) 14 rows, dim
  3 → 2.**  `module_condition_control()`, run at the top of the battery.

**The finding, now that the instrument is known to work.**  On every truncation
whose labels are lines *of the form* — SO(3) `m ≤ 3/2` at character boxes 2, 4
and 6; PSU(3), SO(5), Sp(2)/Z₂ at their fractional coweights, box 3 — the module
condition adds **0 rows**.  So there

    𝒜(★) for the form  =  𝒜(★) for the cover,

i.e. the wall-nullspace of a truncation generated by the form's own lines already
consists of `Λ_H`-preserving operators, and the D24 distinction — real as a
definition — does not bite on the spanning question.  It bites only when the
ansatz is seeded with a label the form does not admit, which is the control.

That is a measurement on small truncations, not a theorem, and the honest reading
is narrow: it says the two versions of (★) coincide *here*, not that they always
must.

Battery: `experiments/spanning_test.py` plus the oracle described above.

### 5c. The spanning test WITH MATTER — and (★) alone is not the condition there

Every truncation above is **pure gauge**, which leaves the user's own framing
half-tested: *"In concrete examples (quivers etc) the expectation is indeed that
the `⋆` span **is** the BFN algebra.  Our job then is to argue that our `L_{m,e}`
lie in and generate the BFN algebra"* (2026-08-25).  `experiments/spanning_test_matter.py` [retired 2026-09-19 with the type-A keystone; the measurement stands as recorded]
asks it at `(G, N)`.

**What makes it possible: the μ-level stack.**  `MatterWRQTorus.to_family()`
presents a `(G, N)` element as `{μ-level k⃗: WRQTorus}` — a finite stack of *pure*
torus slices, `from_family` inverting it.  A canonical is triangular in it: at
SU(2) + 1 fundamental, `L_{(2,),(0,)}` has levels 0, 1, 2 with cell supports
`{−2…2}`, `{−1…1}`, `{0}`.  And a gauge wall `v^α = 𝖖^{−k}` does not involve `μ`,
so **(★) is block-diagonal in the μ-level** — which is
`matter_star_bubbling.solve_level`'s measured statement, *"(★) holds separately
at every μ-level — measured on the flow's own `RG(a)`, 10/10 levels"*.  (The
module header's "couples the μ-levels" is about the **Q-frame**, after dividing
out `Z`; in the raw `F`-frame the walls are level-diagonal.)  So the pure row
builder applies slice by slice, and the genuinely new condition is the one that
couples the levels: **μ-divisibility**, `F_{m'}(μ⃗) = Σ_k⃗ μ⃗^k⃗ f_{m',k⃗}` divisible
by `Z_{m'}(μ⃗)`.  `divide_by_linear` is linear in the coefficients, so that too
becomes rows in the same nullspace.

**Result — thirteen truncations, 156 nullspace vectors, every one decomposing
exactly, `unverified 0` and 0 errors throughout.  The last two are genuine
rank 3** — C₃ and A₃ root systems, not U(3), which is A₂ (see below):

| theory | basis | (★) + div rows | dim after (★) → after div | decomposes |
|---|---|---|---|---|
| SU(2)+1×2 `m ≤ 1` | 24 | 30 + 6 | 11 → **5** | 5 |
| SU(2)+1×2 `m ≤ 1` dressed | 36 | 58 + 6 | 13 → **7** | 7 |
| SU(2)+2×2 `m ≤ 1` | 36 | 50 + 12 | 15 → **3** | 3 |
| SU(2)+2×2 `m ≤ 1` dressed | 48 | 78 + 12 | 17 → **5** | 5 |
| SU(2)+1×2 `m ≤ 2` | 342 | 876 + 109 | 76 → **28** | 28 |
| SU(2)+3×2 `m ≤ 1` | 60 | 96 + 24 | 21 → **3** | 3 |
| SU(2)+1×Adj `m ≤ 1` | 43 | 68 + 18 | 15 → **4** | 4 |
| SU(3)+1×3 `m ≤ (1,1)` | 99 | 398 + 13 | 11 → **2** | 2 |
| **SU(2)+1×2 `m ≤ 3`** | 1917 | 5798 + 690 | 294 → **89** | 89 |
| **Sp(3)+1×6 `m ≤ (1,0,0)`** (C₃) | 99 | 556 + 13 | 13 → **4** | 4 |
| **SU(4)+1×4 `m ≤ (1,0,0)`** (A₃) | 544 | 4048 + 25 | 13 → **2** | 2 |
| U(2)+1×2 `m ≤ (1,0)` | 2 | 0 + 0 | 2 → 2 | 2 |
| U(2)+2×2 `m ≤ (1,0)` | 2 | 0 + 0 | 2 → 2 | 2 |

**The third column is the finding.**  With matter, **(★) alone is not the
condition**: divisibility removes a further 6 to **205** dimensions, and the
`L`'s span exactly what survives *both*.  The cut **deepens** with the truncation
rather than washing out — 205 of 294 at `m ≤ 3`, against 6 of 11 at `m ≤ 1` —
which is the obvious worry about reading a ratio off a small case.  So the pure statement "the `L`'s span the
(★)-algebra" does not carry over unchanged — at `(G, N)` the object they span is
cut by (★) **and** μ-divisibility, which is the repo's own four-condition table
(`matter_star_bubbling`'s header) showing up as linear algebra.

That column is also the **built-in control**, and it is the lesson of §5b applied
in advance: a divisibility block that cut nothing would be indistinguishable from
a no-op.  The two U(2) rows where it cuts 0 (minuscule `m`, no walls and no
`Z`-factor) are **flagged in the output** as `(★)`-only rather than counted as
evidence for it.

**Two frame errors on the way, both caught by controls rather than by reasoning**
— recorded because each is the kind that reads as a mathematical result:

1. **Divisibility is a condition on `F`, not on the chart.**  Applied to a chart's
   `MatterWRQTorus` residuals it *fails on genuine canonicals* — every cell
   carrying a `Z`-factor reports `divisible=False`.  The reason is that
   `gn_abe_kalgebra` runs `divide_by_Z_vec` **after** solving and keeps the
   **quotient**, so a chart residual is `Q` with `F = Z·Q`; in the chart frame the
   condition is vacuous by construction and testing it there is dividing twice.
   The ansatz, rows and nullspace all live in the `F`-frame, and `Q` is recovered
   at the end — which is precisely what the divisibility rows guarantee is
   possible.
2. **`divide_by_linear` is not linear as called.**  It reads `n = max(coeffs)` off
   its input, so the map depends on its argument.  Feeding one basis element at a
   time (each at a single μ-level) divides at a different degree per element, and
   superposing the results is not the division of the sum.  Every input is padded
   to the truncation's common box `∏_s [0 … box_s]`; each division drops exactly
   one degree in its own slot, so the box survives the chain.

**Linear quivers, on the same harness.**  `UNQuiverKAlgebra` needed no new
machinery: `QuiverWRQTorus` has the same `{atom: {level: TorusRational}}`
residuals and the same `to_family()` stack over the *product* datum, and its `Z`
factors come from `quiver_wrq_torus._rungs` in the identical
`(slot, weight, 𝖖-shift)` shape — *"links on negative pair-differences,
fundamentals on negative `m_j`"*.  That shape is all the test needs, so it is
injected (`ZData`) and one implementation serves both tiers.  For a pure-node
chain the slots are the **links**, so the μ-levels are link levels.

| quiver | basis | (★) + div rows | dim after (★) → after div | decomposes |
|---|---|---|---|---|
| U(1)−U(1) `m ≤ 1` | 3 | **0** + 1 | 3 → **2** | 2 |
| U(1)−U(1) `m ≤ 1` both nodes | 4 | **0** + 1 | 4 → **3** | 3 |
| U(1)−U(1) dressed | 4 | **0** + 1 | 4 → **3** | 3 |
| U(2)−U(1) minuscule | 2 | 0 + 0 | 2 → 2 | 2 |
| U(2)−U(1) bubbling | 24 | 30 + 6 | 11 → **5** | 5 |
| U(2)−U(1) link-charged | 5 | **0** + 3 | 5 → **3** | 3 |
| U(1)−U(1)−U(1) middle node | 3 | **0** + 1 | 3 → **2** | 2 |
| U(1)³ ends | 4 | **0** + 1 | 4 → **3** | 3 |
| U(1)³ all three nodes | 6 | **0** + 2 | 6 → **4** | 4 |
| U(2)−U(2) bubbling | 30 | 44 + 15 | 12 → **3** | 3 |
| **U(2)−U(2), both nodes charged** | 114 | 296 + 50 | 27 → **7** | 7 |

**The abelian rows are the sharpest form of the finding.**  At U(1)−U(1) there
are no roots, hence no gauge walls, so **(★) contributes zero rows and
μ-divisibility is the entire condition** — and the `L`'s still span exactly what
it cuts.  Whatever "the (★)-algebra" means in the pure tier, at an abelian quiver
the object the canonicals span is cut by divisibility alone.

**And μ-divisibility is not a module condition** (Plan 24 D48).  The natural
reading would be that it is the matter form of Λ-preservation, which would let
the pure story carry over intact.  It is not, and half of that is a theorem:
multiplying by a character changes a numerator and not the denominators, so the
residue of `x·χ_μ` on a wall is `Res_wall(x)·χ_μ|_wall`, which vanishes wherever
`Res_wall(x)` does — and (★) is exactly that vanishing.  So no `χ_μ` imposes
anything beyond (★), the same argument that makes `wall_is_probed` identically
true for every real form (§5b).  Measured against it, divisibility cuts 6, 12, 11
and 48 dimensions at SU(2)+1×2 / +2×2 / +Adj / +1×2 at `m ≤ 2`.  Hence:

> at `(G, N)` the operators preserving the Neumann module form a **strictly
> larger** space than the span of the canonicals; the `L_{m,e}` span a **proper
> subalgebra** of it.

If the `L`-span is the K-theoretic BFN algebra, that algebra is **not** the
Λ-preserving `𝖖`-difference operators at `(G, N)`.  Recorded as a finding for the
user, not resolved.  Battery:
`experiments/matter_divisibility_vs_neumann.py`.

⚠ **The U(2)−U(1) bubbling row is not independent evidence**, and the reason is
worth recording rather than leaving as a coincidence: it reproduces SU(2)+1×2
`m ≤ 1` in all four quantities (24 / 30+6 / 11→5 / 5) because it is the *same
linear system*.  Measured: three cells either way, one positive root either way,
one `Z`-factor at each extremal cell at 𝖖-shift 0 and none at the centre either
way.  That explains the numbers; it is **not** a claim that the two theories are
the same, which has not been checked.

Battery: `experiments/spanning_test_matter.py`, multi-slot native (the level is a
vector; the same slot-by-slot division as `matter_multislot.divide_by_Z_vec`,
reimplemented in `ZData` so that it runs **without the short-circuit** — which is
right for a guard and wrong for row extraction, since the later factors'
conditions would never be seen — and over either tier's `Z`-data).  Scope, stated
narrowly: rank ≤ 2, fundamental and adjoint matter, linear chains of length 2
and 3.  **180 nullspace vectors over 21 truncations, every one decomposing
exactly.**  `SU(2)+1×2 m ≤ 3` is the largest (basis 1917, 294 → 89) and shows the
cut deepening rather than washing out as the truncation grows.  The 3-node chains
add the two-link case, where the μ-level is a 2-vector and the divisibility chain
runs slot by slot.

⚠ `UNQuiverKAlgebra(ranks)` takes **no per-node `Nf`** ("pure-node increment: no
per-node fundamentals"), so fundamentals on a chain are **not a gap in this
test** — they are not on that class.

⚠ **Rank-3 MATTER is still untested, and the two attempts that looked like it
were not.**  Both are recorded because each returned a passing row:

* **U(3)+1×3 and +2×3 at `m ≤ (1,0,0)`** are *trivial*: `(1,0,0)` is minuscule at
  U(3), so the truncation has no wall and no `Z`-factor, `(★)` and divisibility
  both contribute **zero rows**, and the two theories collapse to the same
  `2 → 2`.  The battery's duplicate check flagged the collision on the spot.
* **U(3)+1×3 at `m = (1,0,−1)`** reproduces **SU(3)+1×3 `m ≤ (1,1)`** in all four
  quantities (99 / 398+13 / 11→2 / 2).  `(1,0,−1)` is **traceless**, so it sees
  only the SU(3) factor: measured, 7 cells, `|W| = 6` and 3 positive roots on both
  sides.  It is the rank-2 computation in different coordinates.

✅ **The gap is now closed.**  Sp(3)+1×6 (C₃, 9 positive roots) and SU(4)+1×4
(A₃, 6 positive roots) both build and both hold, with divisibility cutting 9 and
11 dimensions respectively — so **the divisibility cut is not a rank-≤2
phenomenon**; it survives into genuine rank-3 root systems.  What follows is the
record of how the gap was first mis-explained.

🛑 **The first explanation of that gap was wrong.**  It was recorded as
"a genuine test needs non-zero trace and non-minuscule support".  The real reason
is structural and simpler: **U(3)'s root system is A₂**.  Measured — U(3) has
`|Φ⁺| = 3` and `|W| = 6`, exactly SU(3)'s, against SU(4)'s 6 and 24 and Sp(3)'s 9
and 48.  The centre direction contributes no root, so *no* U(3) charge gives a
rank-3 wall structure, traceless or not; U(3) is rank 3 as a group and rank 2 as
a root system.  A genuine rank-3 matter test needs **SU(4) or Sp(3) with
matter** — both of which do build (SU(4)+1×4: 13 cells, μ-levels 0 and 1;
Sp(3)+1×6: 7 cells, μ-levels 0 and 1).

This third collision crossed *runs*, so the in-process duplicate check could not
see it — every row now also prints its raw `sig=(…)` so cross-run repeats are
greppable.

⚠ **Two defects in `LaurentPoly` this probe exposed — fixed** (user: *"if there
is a problem fix it"*).  The visible one was `__add__`'s general path raising
`KeyError`; the real one was underneath it.  `__init__` tested the ORIGINAL
coefficient for zero but stored `int(c)`, so a non-integral coefficient passed
the test and was silently **truncated** — `Fraction(1,2)` became `0`,
`Fraction(3,2)` became `1`.  That is a wrong value, and the stored zero broke the
class's own documented invariant ("a sparse dict of NON-ZERO coefficients"),
which is what the unguarded `del` then tripped over — while the monomial fast
path, which *was* guarded, sailed through.  Same failure class as the
`_norm_weight` truncation that sent SO(5)'s spinor weight `(½,½)` to `v^0`:
`int()` applied where exactness was assumed.  `__init__` now refuses a
non-integral coefficient (this ring is `Z[q,q^{-1}]`), and the two addition paths
agree.  Regression: `tests/test_laurent_poly_add.py`, controlled against the
pre-fix code (4 of its checks fail there).

**The cost wall, measured** (user, 2026-08-25: *"times will dilate"*).  G₂
`m ≤ (1,0)` takes ~20 minutes against seconds for the rank-2 simply connected
cases, and PSU(3) at `m ≤ (2,2)` was **abandoned** after 25 minutes without
finishing — recorded as not-run rather than quietly dropped.  The cost is
dominated by the exact-rational nullspace and by one `decompose` per nullspace
vector, so it grows with the number of cells and with the `𝖖`-window, not with
the rank as such.  Two things would buy headroom before rank 3: the `𝖖`-window can
be collapsed to `qextra = 0` (it plays no role once `decompose` is the oracle),
and the nullspace can be taken mod two primes with exact verification only on the
survivors.

**What is still open.**  Spanning is verified at `m = 0` (a theorem) and in the
four truncations above; it should be run at larger `m`, on genuine quivers, and
with matter before anything is claimed in general.  And spanning implies closure but is not implied
by it, so even a general spanning result would be a route to closure rather than
a restatement of it.

### A refinement that would localize a failure

Rather than one joint truncation, filter by the dominance order on the magnetic
charge and ask, level by level, whether every (★)-element supported at or below
`m` is a combination of canonicals at or below `m`.  That is the peel/induction
argument in linear-algebra form: it turns a single yes/no into a statement about
*where* spanning first fails, if it does.  It is also where the magnetic support
bound would re-enter — that bound is now known to be a consequence of (★)
(`kalgebra.md`; `experiments/support_bound_subsumed.py`), which is what makes the
induction's termination plausible rather than assumed.

### Cost

The machinery exists: the (★) rows are `joint_solve`'s, and the streaming
eliminator returns the rank as `len(piv)`.  What is new is dropping the seed /
W2 / `O(𝖖)` conditions and computing a nullspace dimension instead of a unique
solution.  The scale is the same as a solve at the largest label in `B`, twice
(two primes), plus once more at `pad+1` — so the practical limit is the
elimination cost recorded in the plan's decisions, and the first runs should sit
well inside it (SU(2), SU(3), PSU(3) at small `m`) before anything larger.

## 6. Next steps (ordered)

1. ~~`|N(k)]` and general symmetric test functions~~ — **DONE** (R4b, R6).
2. ~~General G~~ — **DONE** (R7): SU(2) and SU(3) pass, the partner rule is the
   affine Weyl reflection, and the criterion rejected the cone-only builder's
   un-peeled SU(2) seed.
3. ~~**Unknown bubbling, for real.**~~ — **DONE at rank ≤ 2, and now at `e ≠ 0`
   too** (§4c/§4d/§4f): SO(5), G₂ (`e = 0`), SO(7), Sp(6) have no bubbling
   formula in the repo and were recovered from the axioms alone, with
   orthonormality as the independent oracle; B₂ at `e ≠ 0` — the case reported
   unreachable — is now four labels clean.  What remains under this heading:
   **G₂ at `e ≠ 0`**, and rank 3 re-verified under the corrected device.
4. ~~**Is Weyl covariance subsumed?**~~ — **CONFIRMED, see §4g.**
4. **Matter — DEFERRED (user ruling, 2026-07-27):** *"Do not yet add matter to
   the 4d theory.  That is a later discussion though you are not wrong in your
   approach.  We may look at adding boundary chiral matter.  But a step at the
   time."*  The §3 matter probe stands as a recorded first look only; the `Ξ`-frame
   redo is **not** to be pursued until the user reopens it, and boundary chiral
   matter is the flagged eventual direction rather than 4d matter.
5. **Tighten R4.**  Include the conditions coupling two unknown sectors; the
   nullity should drop from 4 to 1 in the multi-interior-sector cases, and the
   claim "null space = lower canonicals" should then be exact rather than
   inclusive.
6. **The `L̃` frame directly** (rather than via the `ρ`-image proxy).

## 7. Cross-references

| topic | doc |
|---|---|
| the atoms `U_m`, `ψ_m`, the cocycle, `ρ`, the §6b pairing | `pun_enriched_qt.md` |
| left/right = `u`/`ũ`, adjointness | `pun_enriched_qt.md` §6e |
| the vacuum-state Schur pairing and the derived weight `w_m` | `aux_vacuum_pairing.md` |
| the matter window `Ξ` (per-cell, derived) | `aux_vacuum_pairing.md` §4m |
| easy-L̃ vs easy-Neumann; the solved `f_m`; Task F polynomiality | `aux_neumann_wavefunction.md` §2l |
| the constructive-build rule and why a solve is barred on this tier | `CLAUDE.md`; `pure_un_joint_fiber_defect.md` |
| non-minuscule / unknown-bubbling groups | `restructuring_plans/24_nonminuscule_groups/` |
| the AbeKAlgebra contract triple and W1/W2 acceptance | `kalgebra.md` §"AbeKAlgebra" |
| the derived ρ-sign `(−1)^{⟨Σ⁺,k⟩}` | §4i; `general_g_joint_bubbling_solver.weyl_vector_rho_sign` |
