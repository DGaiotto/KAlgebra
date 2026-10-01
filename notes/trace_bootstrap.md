# Trace bootstrap — constructing `Tr` from cyclicity + orthonormality

*A per-example recipe and honest harness, not a universal engine.* (User
direction, 2026-06-13: *"I do not know if anything can be done in universal
generality. I need some structure to guide other sessions handling specific
examples."*)  This note gives the mechanism, a runnable harness, and the
per-example checklist of where it closes vs. honest-fails — so a session can
point it at a **specific** KAlgebra and read off what that example needs.

## 0. What this is / isn't

- **Is:** a *constructive forward recursion* that **derives** the elementary
  traces `T_c := Tr(L_c)` from `(multiply, ρ)` + the two trace axioms, order
  by order in `𝖖`, seeded only by the vacuum `Tr(1)`.  It turns `trace` from a
  trusted primitive into a **certified-derived** quantity (`repo_audit.md`
  Level D, step 3).
- **Is not:** a universal engine.  It is a *structured procedure* plus an
  *honest harness* (`experiments/trace_bootstrap.py`) that reports, per
  example, which traces it pins (and to what `𝖖`-order) and where it
  **stalls** — never fabricating an unreached trace.
- **Distinct from** `trace_uniqueness.py` (the global per-order linear solve
  that tests *uniqueness up to rescale*, taking the known `Tr` as input,
  Goal 2.12 — this note is the **producer**, that the **verifier**) and from
  the **trace miracle** (`experiments/u1_square_trace_recursion.py`, Goal
  2.11 — the *separate* stabilization that pins `Tr(1)` itself, §6).

## 1. The two trace axioms, in elementary-trace coordinates

Let `T_c := Tr(L_c)`; structure constants `C^c_{ab}(𝖖)` from `multiply` are
known.  The contract's two trace axioms become:

- **ρ²-cyclicity:** `Tr(L_a L_b) = Tr(ρ²(L_b) L_a)`; equivalently `Tr` is
  constant on ρ²-orbits.  *Homogeneous* relations among the `T_c`.
- **Orthonormality:** `I_{a,b} = Tr(ρ(L_a) L_b) = δ_{a,b} + O(𝖖)`.  The
  *inhomogeneous boundary* that seeds the recursion.

## 2. The sharp mechanism — three moving parts

> The earlier "order-gain isolation" (§3, a degenerate slice) is *one* way the
> recursion can close.  The general engine — the one that closes the
> **infinite Cartan towers** of U(1)-gauged / quantum-group K-algebras — has
> three parts, separating the roles of the two axioms cleanly.

**(A) Seed — from orthonormality.**  Put `b = 1` in `I_{a,b} = δ_{a,b}+O(𝖖)`.
Since ρ permutes labels with `ρ(a)=1 ⇔ a=1`,

```
   Tr(L_c)  =  δ_{c,1} + O(𝖖)        for every canonical label c.
```

So `Tr(1) = 1 + O(𝖖)` and `Tr(L_c) = O(𝖖)` for `c ≠ 1` — the universal `O(𝖖)`
ansatz the recursion starts from.

**(B) Sector projection — from cyclicity with a Cartan element.**  If a basis
element `K` (a "Wilson/Cartan" line) satisfies `ρ²(K) = K` and
`K·L_a = 𝖖^{w(a)}·L_a·K` for a weight `w(a)`, then
`Tr(L_a K) = Tr(ρ²(K) L_a) = Tr(K L_a) = 𝖖^{w(a)} Tr(L_a K)` forces
`Tr(L_a K) = 0` whenever `w(a) ≠ 0`.  Hence **every trace reduces to the
Cartan sector**: the only nonzero elementary traces are the Cartan powers
`Tr(K^n)`.  (Net-charged / monopole labels have vanishing trace.)

> *Abelian-Cartan caveat.*  (B) needs a **group-like** Cartan with a 𝖖-weight
> `w(a)`.  For a **non-abelian gauge** Cartan it fails: in pure SU(2) the
> Wilson characters `W_n = χ_n` fuse by Clebsch–Gordan and are ρ-*fixed*, so
> there is no weight to project with and the monopole ('t Hooft) sector has
> **nonzero** traces.  The seed sector is then the whole Wilson (gauge-Cartan)
> Schur index, and cyclicity propagates it to the monopole sector (§5,
> pure SU(2)).

**(C) Order-raising — the cyclicity recursion with a unit prefactor.**  Apply
cyclicity to `Tr(K^n·u₊u₋) = Tr(ρ²(u₋)·K^n·u₊)`, where `u₊u₋` is a monopole
("`E·F`") product that closes on the Cartan tower.  Expanding both sides in
the basis and keeping the surviving (Cartan) terms gives a relation in which
`Tr(K^n)` recurs on both sides; after rearrangement its net coefficient is a
**unit** `(1 − 𝖖^{2n})` (monic at `𝖖⁰`), and every *other* term carries an
explicit **positive** `𝖖`-power.  Therefore neighbours known to `O(𝖖^k)` pin
`Tr(K^n)` to `O(𝖖^{k+1})`: a single increasing-`𝖖`-order sweep fills the whole
tower.  The `n=0` relation degenerates, so **`Tr(1)` is left free** — the
contract's `1 + O(𝖖)` rescale (§6).

`experiments/trace_bootstrap.py::cyclicity_tower_bootstrap(A, cartan,
cartan_index, monopoles, K)` runs exactly this: it builds the per-`n`
relation, auto-rescales it to the `(1−𝖖^{2n})` form, and forward-sweeps.

## 3. The order-gain isolation step (the degenerate slice)

When ρ has *finite* order (finite type) the recursion can also close
label-by-label without the Cartan projection.  Expand `ρ(L_a)·L_b = Σ_c
C^c L_c`; if one `c*` is the **unique most-𝖖-negative** term
(`C^{c*}=u·𝖖^{−m}+…`, `u=±1`, `m≥1`, others valuation `≥ −m+1`), then

```
   T_{c*}  =  −u·𝖖^m·Σ_{c≠c*} C^c·T_c   +   O(𝖖^{m+1}),
```

so others known to `O(𝖖^{m−1})` pin `c*` to `O(𝖖^m)` (the `𝖖^m·I_{a,b}`
remainder is inert).  `bootstrap_traces` / `verify_bootstrap` run this as a
vacuum-seeded greedy fixpoint with honest stalls.  It is a *special case* of
§2 (orthonormality boundary + cyclicity), and it does **not** handle infinite
towers — see the Sqed1 contrast in §5.

## 4. The canonical template — `U_𝖖(𝔰𝔩₂)` (paper example, `main.tex`)

The cleanest instance is the SU(2)-flavoured `K_𝖖`-algebra of SQED₂
(`U(1)` gauge + `T*ℂ²`), isomorphic to the central quotient of `U_𝖖(𝔰𝔩₂)`
(this is also the seed of Goal 1.7.a, *quantum groups as KAlgebras*).
Generators `K^±, E, F`, characters `χ_k` (SU(2) flavour):

```
   K E = 𝖖^{−2} E K ,   K F = 𝖖^{2} F K ,
   E F = χ₁ + 𝖖 K + 𝖖^{−1} K^{−1} ,   F E = χ₁ + 𝖖^{−1} K + 𝖖 K^{−1} ,
   ρ(K)=K^{−1} ,  ρ(E)=𝖖^{−1} F K^{−1} ,  ρ(F)=𝖖 K E ,   ρ fixes χ_k.
```

ρ is **Lusztig's braid symmetry — infinite order** (it covers the Weyl
reflection `K↦K^{−1}` but acts as the braid group `ℤ`; *false friend alert:
not a finite ρ like the pentagon's*).

- **(B)** `Tr(L_a K) = Tr(ρ²(K) L_a) = Tr(K L_a)` ⇒ net-`E`/`F`-charged
  monomials have zero trace; survivors are the `Tr(K^n)`, collected by
  `G(x,μ) = Σ_n Tr(K^n) x^n = (𝖖²;𝖖²)_∞² · E_𝖖(μx)E_𝖖(μ^{−1}x)
  E_𝖖(μx^{−1})E_𝖖(μ^{−1}x^{−1})`, `E_𝖖(x)=(−𝖖x;𝖖²)_∞^{−1}`.
- **(C)** cyclicity on `Tr(K^n E F)=Tr(ρ²(F) K^n E)` gives
  `𝖖^{2n}(Tr K^n + 𝖖χ₁ Tr K^{n+1} + 𝖖² Tr K^{n+2}) = Tr K^n + 𝖖χ₁ Tr K^{n−1}
  + 𝖖² Tr K^{n−2}`, i.e. for `n>0`
  ```
   (1 − 𝖖^{2n}) Tr K^n = 𝖖^{2n+1} χ₁ Tr K^{n+1} + 𝖖^{2n+2} Tr K^{n+2}
                         − 𝖖 χ₁ Tr K^{n−1} − 𝖖² Tr K^{n−2}.
  ```
- **(A)+(C):** *"Given any `Tr 1 = 1 + O(𝖖)`, iterate the recursion from the
  `O(𝖖)` approximation `Tr K^n = O(𝖖)` to a unique solution at `O(𝖖²)`, then
  `O(𝖖³)`, …"* — and `G(x,μ)` is such a solution.

**Verified (2026-06-13), `experiments/sqed2_uqsl2_recursion.py`.**  Carrying
`χ₁ = μ + μ⁻¹` through the 4-term recursion above, seeded only by
`Tr K⁰ = [x⁰]G`, reproduces the **full μ-refined** `G(x,μ)` for every `Tr K^n`
(e.g. `Tr K¹ = −𝖖·χ₁ − 𝖖³·χ₃ − …`, `Tr K⁰ = 1 + 𝖖²(χ₂ − 1) + …`,
Weyl-symmetric; the `−1` is the U(1) vector multiplet's `(𝖖²;𝖖²)²_∞` — this
line printed `1 + 𝖖²χ₂ + …` until 2026-09-21, a slip against both the closed
form and the live trace below).
**Triple validation:** the closed-form `G`, the repo realisation
`legacy/sqed2_su2_over_pure.py::Sqed2SU2OverPure.trace` [in `legacy/` since
2026-09-19, #1510; the measurement stands, and the live SQED₂ is
`GNAbeKAlgebra(u_n(1), (1,), nf=2)`, whose Wilson traces reproduce `G(x,μ)`
exactly through `𝖖⁹` at `n = 0, ±1, 2` after `base_change(un_to_sun_hom(2))`
— 2026-09-21] (computed
independently via the FS pairing), and the recursion all agree (`Tr K^n`
through `𝖖⁷`, `n=0,1,2`, exactly).  This is the flavoured (non-trivial-`R`)
extension of the mechanism — and the first `U_𝖖(𝔤)`-as-KAlgebra data point
for Goal 1.7.a.

## 5. Worked examples in the repo (actual harness output)

**SQED₁ / U1Square** (`Sqed1KAlg`, U(1)-gauged, the same shape as §4 with
trivial flavour).  Labels `(m,n) = (magnetic, Coulomb)`; `ρ(m,n) =
(−m, −n−max(m,0))`, so `ρ²(m,n)=(m,n+m)` and the **`m=0` tower is the
ρ²-fixed Cartan sector** with `ρ(0,n)=(0,−n)` (the `K↦K^{−1}` analog) and
monopoles `u_± = (±1,0)`.  Trace vanishes off `m=0`.  The §2(C) recursion is

```
   (1 − 𝖖^{2n}) Tr(0,n) = 𝖖^{2n+1} Tr(0,n+1) − 𝖖 Tr(0,n−1)      (n ≥ 1)
```

(two terms instead of four — SQED₁ vs SQED₂), and `cyclicity_tower_bootstrap`
**closes the whole tower from the single seed `Tr(1)`**:

```
tower -14..14  seed=Tr(1)  K=24  (28 relations pivoted)
  Tr(K^1) == A.trace through q^22   …   Tr(K^8) == A.trace through q^8   [all OK]
==> TOWER CLOSES FROM Tr(1) ALONE
```

**Correction to an earlier finding (2026-06-13):** the *isolation* harness
(§3) on SQED₁'s full label set reports stalls — the `+v` tower and the
magnetic rows `(m,0)` — which I first read as "one seed per magnetic sector."
That was an **artifact of the wrong mechanism**: with the §2 Cartan projection
the magnetic traces are *zero*, and the single Cartan tower closes from `Tr(1)`
alone.  SQED₁ works exactly as `U_𝖖(𝔰𝔩₂)` — one seed, no extra.

**Pentagon** (`A_𝖖([A₁,A₂])`, finite type, finite ρ).  The §3 isolation step
already closes the *whole* window from the vacuum seed (30/30 derived
`== A.trace`, orders growing with cone depth, no stalls) — the finite-type
case where every trace is reached label-by-label.

**The whole `A1A2k(k)` golden-standard family** (`A_𝖖([A₁,A_{2k}])`,
pentagon/heptagon/nonagon = k=1,2,3; `experiments/a1a2k_trace_bootstrap.py`).
Same §3 mechanism, run on the *standalone cone class* `A1A2kKAlg(k)` (spine-free)
and parametrically in `k`: the producer derives every deg-≤2 cone-window trace
from `Tr(1)` (45 / 693 / 5667 traces, 0 stalls, 0 conflicts), each `== A.trace`
= the closed-form M(2,2k+3) Andrews–Gordon character.  Paired in the same driver
with `trace_uniqueness_proofs.prove_uniqueness` (the verifier: solution space =
the `1+𝖖·Q[[𝖖]]` rescale line, 0 cyclicity relations / free seeds), this is the
"explicit trace = unique solution of cyclicity + orthonormality" demonstration
on the reference-grade family.  Pinned in `tests/test_a1a2k_trace_bootstrap.py`.

**Pure SU(2)** (`pSU2KAlgebra`, the **non-abelian-gauge** regime;
`experiments/psu2_cyclicity_bootstrap.py`).  Labels `(m,e) = (magnetic,
electric)`; Wilson `W_n=(0,n)=χ_n` is **ρ-fixed** and **CG-fusing**
(`W_1 W_n = W_{n+1}+W_{n-1}`), the 't Hooft `H_0=(1,0)` has `ρ²: e↦e−8`
(infinite order), and crucially `Tr(H_0)=−𝖖+𝖖⁵−𝖖⁷+… ≠ 0` — the sector
projection (B) does **not** apply.  The bootstrap is two-layered: the **seed
sector** is the non-abelian Wilson Schur index `Tr(W_n)=[v^n]F−[v^{n+2}]F`,
`F(v)=(𝖖²v²;𝖖²)_∞²(𝖖²v⁻²;𝖖²)_∞²(𝖖²;𝖖²)_∞²`, and the **monopole traces follow
by cyclicity** — e.g.

```
Tr(H_0) = −𝖖⁻¹·Tr(W_0) + ((2𝖖²−1)·Tr(W_2) + Tr(W_4)) / (2𝖖³·(1−𝖖²))
```

(from `Tr(H_{−a}H_a − 𝖖^{2a}H_0²)` cyclicity at `a=1,2`; the `1/(1−𝖖²)` is the
unit-prefactor order-raising, the same engine as §2(C)).  Verified: the bridge
gives `−𝖖+𝖖⁵−𝖖⁷`, `== A.trace(H_0)`.  The closed-form dyon bridges (m=1,2,3)
are in `implementations/pure_su2_h_trace.py`.

## 6. The synthesis (links Goal 2.11 ↔ 2.12)

The recursion + the §2(A) seed pin every higher trace **given `Tr(1)`**.
Since `Tr_canonical(1) = 1 − 𝖖² + …` is a unit in `Q[[𝖖]]`, fixing the single
series `Tr(1)` forces `f = 1` in the contract's "`Tr` up to `1 + O(𝖖)`
rescale".  Therefore:

- **Goal 2.12 (uniqueness):** bootstrap + orthonormality pin `Tr` *up to*
  `Tr(1)`'s own tail = the documented `1 + O(𝖖)` rescale line (what
  `trace_uniqueness.py` measures globally; the `n=0` relation degenerating in
  §2(C) is the same freedom, seen in the recursion).
- **Goal 2.11 (the miracle):** the *separate* fact that running the recursion
  to large depth and normalizing by `𝖖^{emin}` reconstructs `Tr(1)` **itself**,
  removing even the rescale — which the two axioms alone do **not** (the
  uniqueness result *is* a 1-parameter family).

So: **this recipe = the constructive half; `trace_uniqueness.py` = the
global-solve confirmation; the miracle = the extra input that pins the seed.**

## 7. Per-example checklist (run it on your KAlgebra)

1. **Cartan element(s) `K`.**  Find a basis element with `ρ²(K)=K` and a
   `𝖖`-weight grading `K L_a = 𝖖^{w(a)} L_a K` (often `ρ(K)=K^{−1}`).  Its
   powers `K^n` are the surviving **tower**; `cartan_index(L)` returns the
   tower index or `None` (the §2(B) projection).  *Multiple* commuting Cartans
   ⇒ multiple towers (each its own `Tr(1)`-type seed).
2. **Monopole probe `u₊u₋`.**  The `E·F`-analog whose product closes on the
   tower; this generates the §2(C) recursion.
3. **Run** `cyclicity_tower_bootstrap` (seed `Tr(1)`); read the pinned orders
   and any `MISMATCH` (a wrong Cartan/probe surfaces here, never silently).
4. **Where it breaks (verify against the repo, per false-friends):**
   - *No unit prefactor* — if the diagonal `(1−𝖖^{2n})` is not a unit (or the
     auto-rescale leaves a non-positive cross-term), the relation does not
     pivot; the harness skips it (reports fewer "relations pivoted").
   - *Ties / coupled blocks* (§3) — two new labels at one depth need a small
     linear solve, not a clean isolation.
   - *Bound states* — `q-order(δ)=h(δ)+binding(δ)` (`kalgebra.md`); on chambers
     with bound states the valuation can mislead (heptagon `#309/#310`).
   - *Infinite ρ²-orbits* — handled here by the Cartan projection; without it,
     `_canonical_rho2_orbit_rep` needs the closed-form drift-quotient.
   - *Exact arithmetic, truncate last* — the `𝖖^{−m}`/negative powers are large
     and cancel exactly; never start in truncated series.

## 8. Toward higher rank — `U_𝖖(𝔰𝔩₃)` (mapped, 2026-06-13)

The next rung (Goal 1.7.a) is the rank-2 quantum group `U_𝖖(𝔰𝔩₃)` =
`QuiverOverPure((1,2), Nf=(0,3))` (the U(1)-U(2)+3fl quiver,
`implementations/uq_sl3_quiver.py`; `verify_uq_sln_relations` certifies the
Chevalley structure).  Structural mapping done this session:

- **Rank-2 Cartan tower.**  Two Cartans `w₁,w₂` with `ρ(w_a)=w_a⁻¹`; the
  surviving sector is `Tr(w₁^a w₂^b)`, μ-refined (SU(3) flavour in the values),
  seeded by orthonormality `Tr=δ_{(a,b),0}+O(𝖖)`.
- **Two `E_aF_a` probes.**  Node 1 (U(1)) gives a clean SQED-shaped
  `E₁F₁ = 1 + 𝖖(…) + 𝖖²(…)`; node 2 (U(2)+3fl) brings in **SU(3) matter** (the
  `e_k` characters + a W-boson term) via the `W₂` closure — the two nodes
  couple (the T[SU(N)] cascade).
- **The new ingredient is non-abelian gauge** (the U(2) node): the surviving
  magnetic-0 sector is the full **non-abelian Wilson character lattice**
  (it includes W-boson directions like `(((0,),(0,)),((1,−1),(0,0)))`), not a
  simple abelian tower — `section_decompose` cleanly folds flavour into
  μ-coefficients, but the tower itself is rank-2 non-abelian.

So `U_𝖖(𝔰𝔩₃)` is a **multi-step build**: the rank-2 non-abelian Wilson recursion
+ the matter-coupled node-2 closure.  Pure SU(2) (§5) is the clean isolation of
the non-abelian-Wilson ingredient and the recommended stepping stone.  The
generic flavoured cyclicity-tower solve on the magnetic-0 sector (μ carried,
two probes, order-by-order from `Tr(1)`) is the natural instrument; building and
certifying it against the `QuiverOverPure` trace is the pickup-ready follow-up.

## 8b. The cone-fan unification — regime = lineality rank (2026-06-22)

A structural synthesis that removes the per-example dispatch and answers "how to
talk about the canonical basis generically" (user, 2026-06-22: *the Cartan
"just looks like a ray generator shared by many cones"* — exactly right).

**The statement.**  On a cone-presented `KAlgebra`, **the trace lives only on the
`ρ²`-fixed commutant** — the sublattice spanned by the `ρ²`-fixed mult-gens that
q-commute with everything, i.e. the **lineality of the cone fan** (the rays
shared by *all* cones).  Everything transverse is killed by the universal
projection already in `cone_data.py::trace_vanishes_by_rho2_fixed_factor`:
for a `ρ²`-fixed factor `g`, `(1 − 𝖖^{−2·net_c(g)})·Tr(g·O)=0` with
`net_c(g)=Σ_h powers[h]·cocycle(g,h)`, so `Tr=0` unless the net cocycle (= the
§2(B) q-weight `w`) vanishes.  Thus the "Cartan" is **not a new abstraction**:
it is a lineality ray, the q-weight is `cocycle(g,·)`, and the sector projection
is the existing rule.

**The lineality rank is the regime** — one pipeline, no dispatch fork:
`reduce (cocycle/cross_product) → restrict to lineality → pin from Tr(1)`.

| lineality rank | surviving sector | producer | examples |
|---|---|---|---|
| **0** | finitely many seeds | orthonormality solve (§3) | pentagon, heptagon, e6/e7/e8, a3/a5/a7, a1d3–a1d8 |
| **1**, abelian | a 1-D torus tower `{Tr(Kⁿ)}` | `(1−𝖖^{2n})` recursion (§2) | U1Square (closes) |
| **1**, multi-letter | tower + extra mag-0 generators | §2 + sandwiched relation | U1Hex/Octagon/Decagon (Goal 2.11 B-family) |
| **r > 1**, abelian | lineality lattice `Zʳ`, `Tr(w₁^a w₂^b…)` | multi-index recursion | `U_𝖖(𝔰𝔩₃)` (§8); longer quivers |
| any, **fusing** | non-abelian Wilson character algebra | Wilson-Schur seed + propagation (§5) | pure SU(2) |

Verified at catalog scale (`experiments/smart_trace_bootstrap.py::sweep_regimes`):
**every** finite-zoo entry classifies rank 0; **every** U(1)-polygon rank 1 —
the empirical backbone of "finite type ⇔ trivial lineality".  Caveat (per the
false-friends discipline): the clean torus picture is the *abelian-Cartan* case;
**non-abelian gauge** (pure SU(2), a `CharacterCone`) fuses by
Clebsch–Gordan, carries no q-weight, and its commutant is a character
algebra — §2(B)'s caveat and §5.

**Tooling** (`experiments/smart_trace_bootstrap.py`).  `cone_cartan_commutant(A)`
reads the lineality rays + transverse monopole pool off the fan;
`cone_tower_inputs(A, comm)` derives `(cartan, cartan_index, monopole)` for the
rank-1 torus case (the monopole = the transverse `cross_product` pair closing on
the tower — no scan, no oracle-validation, unlike the cone-less heuristic path);
`trace_regime_report(A)` / `sweep_regimes(...)` emit the classification above.
The smart-bootstrap dispatcher routes a cone algebra with a lineality ray to the
fan-driven `cone_tower` producer (U1Square closes from `Tr(1)` matching `A.trace`
at all orders).  **Open producers**: rank ≥ 2 multi-index; non-abelian/fusing
Wilson-Schur; rank-1 multi-letter (U1Hex — closed in principle by
`experiments/hexagon_closure_forward.py`, not yet wired generically).

## 9. Pointers

- **Per-example status ledger** (which traces are recovered up to `Tr(1)` and
  which are good future targets): [`trace_bootstrap_status.md`](trace_bootstrap_status.md).
- Harness + worked examples: `experiments/trace_bootstrap.py`
  (`cyclicity_tower_bootstrap` = §2; `bootstrap_traces` = §3);
  `experiments/sqed2_uqsl2_recursion.py` (the flavoured `U_𝖖(𝔰𝔩₂)` recursion
  recovering `G(x,μ)`, triple-validated); `experiments/psu2_cyclicity_bootstrap.py`
  (pure SU(2), the non-abelian-gauge instance; `implementations/pure_su2_h_trace.py`
  for the closed-form dyon bridges); `experiments/hexagon_cyclicity_bootstrap.py`
  (U1Hex — the open Goal-2.11 case: an *expanding* multi-letter mag-0 seed
  hierarchy, Z₃-reduced, ρ²-shear-tied); `experiments/u1_square_trace_recursion.py`
  (the miracle side, Goal 2.11).
- Uniqueness (global solve): `trace_uniqueness.py`,
  `trace_uniqueness_proofs.py`, `trace_uniqueness_flavoured.py`;
  results in `finite_type_kalgebras.md` §5.
- Theory: `main.tex` §"`U_𝖖(𝔰𝔩₂)` quantum group as a flavoured `K_𝖖`-algebra"
  (the §4 template); `kalgebra.md` (the trace contract; the "ρ is
  piecewise-linear" + bound-state cautions).
- Goals: `RESEARCH_GOALS.md` §2.11/§2.12, §1.7.a (quantum groups);
  `repo_audit.md` Level D step 3.
