# SU(3)-flavour Layer-1 / multiply defect (the "ρ conjugates fugacities" bug)

> **FULLY RESOLVED, BPS-FREE (2026-06-20).**  The remaining open item below
> — a *BPS-free* correct trace of a **product** (the premature-symmetrization
> of non-self-dual content) — is closed.  `implementations/sl3_su3_traces.py`
> supplies a **fugacity-level multiply** (`fug_multiply`: the cone multiply
> with SU(3) Clebsch–Gordan replaced by Cartan-weight `zmul`, so weights — not
> characters — flow through the product) and a product trace = fugacity-
> multiply → single-label trace per canonical monomial → Weyl-symmetrize the
> **total** only.  `SU3ADKAlg.{trace,inner_product,trace_word}` now route here
> and touch **no** BPS/RG engine.  Validated: `Tr(T₀·T₂)=Tr(T₀²)` and
> `Tr(D₀·D₂)=Tr(D₀²)` (ρ²-cyclicity), diagonal/off-diagonal `inner_product`,
> and direct vs BPS on `T₀·T₁`, `T₀·D₁`, `D₀·D₁`.  Tests:
> `test_su3_ad_kalg.{test_trace_word_cyclicity,_cyclicity_D,_is_bps_free}`,
> `test_sl3_su3_traces`.  (The character-level `trace_layer1` is kept for
> inspection; the fugacity path is the production one.)


**Status:** TRACE RESOLVED (2026-06-13).  The product trace is correct via
`trace_word` (U(1)²-routed, **adaptive per-term margin**, Weyl-symmetrize the
total); audited == the U(1)² reference on all generator pairs.

**`_tt_dist2` coefficient fix — IMPLEMENTED (2026-06-13).**  The defect was
localized to the adjoint-generating relation `_tt_dist2` (T·T distance-2): a
**wrong coefficient**.  Its gauge-0 content was `2·χ(1,1)+(𝖖⁻²+4+𝖖²)` but the
truth (U(1)² reference) is six root-weights + singlet `= χ(1,1)+(𝖖⁻²+2+𝖖²)` —
it **over-counted the adjoint by `χ(1,1)+2`** (the six roots are `χ(1,1)−2` as
a *virtual* character, not `2·χ(1,1)`).  D·D and T·D distance-2 are clean
(single fundamental Weyl orbit).  Corrected in `_tt_dist2`
(`(0,{},(1,1),2)→1`, `(0,{},(0,0),4)→2`); this makes the cone `multiply`
trace-correct.  Verified: `Tr(T₀·T₂)=Tr(T₀²)`, the `trace_word`/cyclicity/
R-linearity tests, the palindromic test (updated), and the su3→su2u1/a1d4
dependents all pass (SU3Ad 15/15).  Character-level, BPS-free — vindicating
"R(SU(3)) is uniform; characters are fine".

**Cascade handled.**  Correcting `_tt_dist2` changed the cone `multiply`, so
the iso `cert` (`test_iso_matches_bps_products`) now diverges from `to_R_form`
on exactly the T·T distance-2 gauge-neutral term — because `to_R_form`
DOUBLE-COUNTS the adjoint there (returns `2χ(1,1)+4` for the trace-correct
`χ(1,1)+2`).  The cert subtracts this known `to_R_form` fold artifact for the
T·T distance-2 pairs (documented inline), validating the corrected cone
against the real algebra rather than the double-counting fold.

**Remaining (BPS-free reduction only).**  The primary trace paths are correct:
`trace()` (single-label, R-linear, Layer-2 value from BPS — the allowed
exception) and `trace_word` (U(1)²-routed, adaptive margin).  The *fully
BPS-free* `trace_layer1` reduction additionally needs the `Tr(D_odd)=⋆Tr_D`
tile-parity in its base case (the coefficient fix it inherits from `_tt_dist2`
is necessary but not sufficient without parity — verified).  That is a
self-contained follow-on; it does not affect the class's (now correct)
multiply or trace.
**Reporter:** session investigating SU3Ad trace; user (D.G.) co-diagnosed
("ρ inverts flavor fugacities … latent bug in general layer 1 … only SU(2)
flavour is safe").

> **ROOT CAUSE (the real one, 2026-06-13).**  Most of the apparent
> cyclicity *violations* and trace *blow-ups* chased during this investigation
> were **finite-cutoff truncation artifacts**, not algebra defects.  BPS
> products carry **Laurent `𝖖⁻ⁿ`** coefficients, so a term `𝖖⁻ⁿ·L_γ` feeds the
> `𝖖^{≤K}` output only through trace orders up to `K+n`; a **fixed** trace
> cutoff silently drops those contributions and breaks cyclicity.  With an
> **adaptive per-term margin** (trace each term to `K − min(𝖖-powers) + buffer`)
> the U(1)² substrate `_bps` is cyclic and `Tr(T₀·T₂)=Tr(T₀²)={𝖖²:1,𝖖⁸:χ(1,1)}`
> exactly — confirming the user throughout: R(SU(3)) is uniform, characters are
> fine, the algebra is sound.  Earlier claims to the contrary
> ("characters can't represent the content", "multiply is wrong", a
> fugacity-presentation rework) were **mistaken** and are retracted.

## RESOLUTION (2026-06-13, user's principled argument)

**The principle (user, decisive).**  A conventional BPSKAlgebra whose flavour
is the **U(1)² Cartan** of SU(3) certainly exists and satisfies every axiom.
The SU(3)-flavoured class is just its **Weyl-covariant packaging**, so it
*cannot* fail an axiom — any failure is a packaging defect.  This U(1)²
algebra is literally `SU3BPSKAlgebra._bps` (the abelian substrate under the
SU(3) engine).

**The actual bug = premature SU(3)-symmetrization.**  The trace is
R(SU(3))-**linear** (`Tr(χ_μ·1)=χ_μ·Tr_1` holds for *every* rep — verified by
tracing full reps), so the earlier "non-linearity" was an artifact of
`_std_to_bps_label` standing a whole rep in for its single highest weight.
The real defect: a product of flavour-trivial generators develops genuinely
**non-Weyl-symmetric** fugacity content; collapsing it to an SU(3) **character**
*per intermediate term* (as `from_abelian`/`to_R_form`, and the cone relation's
`2·χ(1,1)`, do) symmetrizes too early — wrong for non-self-dual content,
invisible to the character-level cert.  SU(2) flavour is safe (self-dual).

**The fix (applied).**  Work at the U(1)² fugacity level, Weyl-symmetrize to
SU(3) characters only on the *total*:

1. `SU3ADKAlg.trace` — R(SU(3))-linear flavour stripping:
   `Tr(L_{(tile,a,b,p,q)}) = χ_(p,q)·Tr(L_{(tile,a,b,0,0)})`, the gauge
   monomial being flavour-trivial (always Weyl-clean — kills the lossy
   transport *and* the flavoured-window trapezoid).  Regression:
   `test_trace_R_linear_on_flavour`.
2. `SU3ADKAlg.trace_word(factors)` — multiply in the certified
   `SU3BPSKAlgebra`, **lift the whole product to the U(1)² substrate**, trace
   at the fugacity level, Weyl-symmetrize the total.  Verified `Tr(T₀ⁿ)`,
   `Tr(D₀ⁿ)` match the single-label trace and ρ²-cyclicity holds
   (`test_trace_word_matches_single_label`, `test_trace_word_cyclicity`).

**Still character-level (outstanding).**  The BPS-free `trace_layer1` and the
cone `multiply` (`_tt_dist2` et al.) emit SU(3) characters and so mis-handle
non-self-dual product content (e.g. `2·χ(1,1)` traces as `2χ(1,1)·Tr_1`
instead of the fugacity content).  They are correct at the character level
(the cert passes) but not for the trace of products.  A BPS-free correct
version needs the same fugacity-level treatment (present flavour on the U(1)²
Cartan, ρ = `z↦z⁻¹`, symmetrize at the end).

---

## ORIGINAL DIAGNOSIS (kept for the record; superseded by the RESOLUTION above)

> **Terminology correction (user, 2026-06-13):** drop the "weight lattice vs
> character" framing — it is misleading.  *Inverting a fugacity is the same
> as dualizing the representation labelling the character* (`3 ↔ 3̄`, i.e.
> `⋆ : χ(p,q) ↦ χ(q,p)`).  ρ acts on flavour by this rep-dualization.  The
> issue below is therefore about tracking `3` vs `3̄` distinctly under ρ —
> which the Weyl-symmetric character presentation cannot do for non-self-dual
> content.

## DEFINITIVE CONCLUSION (2026-06-13)

The defect is **not** a localizable bug in `_trace_reduce_word` that a small
patch fixes.  It is that the **character (Weyl-symmetric) presentation of
flavour is too coarse to compute the ρ²-twisted trace** for non-self-dual
flavour:

* `multiply` is correct at the **character** level — `test_iso_matches_bps_
  products` passes (12/12) — but **wrong at the fugacity level**.  BPS-native
  `T₀·T₂` carries the genuine non-self-dual pair `(0,0,-2,-1):1 +
  (0,0,-1,-2):1` (a `3 + 3̄`); the standalone collapses it to `2·χ(1,1)`
  (`(0,0,-2,-1):2`).  These agree only after Weyl re-symmetrization.
* The **trace needs the fugacity-level data**.  Solving for the elementary-
  trace reduction of `Tr(T₀²)` that matches the known truth `𝖖²+χ(1,1)𝖖⁸`
  shows **no** ⋆-reading of the character relation works — the discrepancy
  is dominated by the relation's *constant* terms (`2·χ(1,1)`, `q⁻²+4+q²`),
  which are simply the wrong objects to trace.  So even a parity/⋆-correct
  `_trace_reduce_word` over the existing tables cannot succeed.
* The certificate is blind to this: `bps_form` runs `to_R_form`, which
  re-symmetrizes BPS weights into characters before comparing — exactly the
  step that erases the `3` vs `3̄` distinction.
* The BPS engine is the trusted oracle and traces *full* products correctly,
  but **cannot trace an individually non-symmetric flavour term** ("from_
  abelian: input not S₃-symmetric") — the conjugate pieces must be combined
  first.

**Implication for the fix:** flavour content must be carried at the fugacity
level (track `3` vs `3̄` / weights, ρ = rep-dualization) through both
`multiply` and the trace, collapsing to Weyl-symmetric characters only at the
final elementary-trace readout.  That is a contract-surface change to the
`RKAlgebra` "free over `R(SU(3))`" presentation, and the cert must be
re-stated at fugacity resolution.  SU(2)-flavour examples are unaffected
(self-dual: character = its own dual).

## One-line statement

The standalone SU(3)-flavoured algebra (`SU3ADKAlg`) represents flavour
content as **Weyl-symmetric SU(3) characters** `χ_(p,q)`, but the genuine
product flavour content lives in the **weight lattice** (Cartan fugacities)
and is *not* Weyl-symmetric until the trace.  Because `ρ` acts as the
charge-conjugation `⋆ : (p,q)↦(q,p)` (a Weyl element) on flavour, a
weight pair `w + ⋆w` is **not** equal to `2·χ_hw`.  The distance-2 `T·T`
relation `_tt_dist2` makes exactly this false identification, so both
`multiply` and `trace_layer1` are wrong for non-self-dual flavour.
SU(2) flavour is safe (all SU(2) irreps are self-conjugate).

## Concrete reproduction (the smoking gun)

`implementations/su3_ad_kalg.py`, generators `T_0=(0,1,0,0,0)`,
`T_2=(4,1,0,0,0)`.

```
standalone  multiply(T_0, T_2)  transported to BPS labels:  (0,0,-2,-1): 2,  (0,0,-1,-2): 0
BPS-native  multiply(T_0, T_2):                              (0,0,-2,-1): 1,  (0,0,-1,-2): 1
```

`(-2,-1)` and `(-1,-2)` are a ⋆-conjugate weight pair.  The standalone
encodes the adjoint piece as the term `(0, {}, (1,1), 2)` = `2·χ_(1,1)`
in `_tt_dist2`; `_std_to_bps_label` maps `χ_(1,1)` to its single highest
weight `(-(p+q),-p) = (-2,-1)`, giving coefficient **2** there and **0**
at the conjugate.  The true (BPS) content is `(-2,-1):1 + (-1,-2):1`.

Distance-1 (`T_0·T_1`) is **identical** to BPS — the defect is specific
to distance-2 (and presumably higher) where the adjoint / non-self-dual
content appears.

## Downstream consequence (the trace)

`Tr(T_0^2)`:
* truth (single-label BPS `S.trace((0,2,0,0,0))`): `𝖖^2 + χ(1,1)·𝖖^8`.
* standalone reduction (`trace_layer1`, or per-term from the standalone
  multiply): excess of `2 + χ(1,1)` at `𝖖^0` (and wrong throughout).

`Tr(D_0^2)` and all distance-1 / D-channel reductions are **correct**
(verified: `Tr(D_2·D_0) == Tr(D_0^2)` to `𝖖^9`), because the D-relations
and `_tt_dist1` do not hit the bad `2·χ` identification.

## Two genuinely distinct elementary traces, collapsed to one

Independent of the multiply bug, the ρ²-twisted trace splits the D-orbit
by **tile-index parity** (`(tile//2) % 2`):

```
Tr(D_0) = Tr(D_2)               =: Tr_D            (tiles 0,4)
Tr(D_1) = Tr(D_3) = ⋆ Tr(D_0)   =: Tr_D^odd        (tiles 2,6)
```

`Tr_D^odd` is the **⋆-dual** of `Tr_D` (χ(1,0)↔χ(0,1), χ(2,1)↔χ(1,2), …).
`Tr_T` is tile-independent only because it is ⋆-self-dual (only χ(n,n)
and integers appear).  `_trace_reduce_word` (su3_ad_kalg.py:617) emits a
single `('Tr_T',)/('Tr_D',)` symbol per letter, with **no** tile-parity /
⋆ tag — so even a multiply-correct reducer would mis-handle odd-tile D's.
A correct reducer must carry `(kind, tile_parity)` and apply ⋆ to the
elementary-trace symbol on odd tiles.

## Why the "certified" iso missed it

`tests/test_su3_ad_kalg.py::test_iso_matches_bps_products` compares
standalone vs BPS **at the character level**: its `bps_form` calls
`B.to_R_form(prod)`, which re-assembles the BPS weight content into
SU(3) characters `(p,q)` before comparing.  That re-symmetrization is
blind to exactly the weight-vs-character distinction that is broken, so
`2·χ(1,1)` and `(w + ⋆w)`-style content can compare equal.  The check
is therefore not ground-truth at the resolution where the bug lives.
**A correct cert must compare at the weight (fugacity) level.**

## Scope

* Confirmed broken: `SU3ADKAlg` (SU(3) flavour) — `multiply` (distance-2
  T·T) and `trace_layer1` (all non-self-dual reductions).
* Conjectured latent in the **shared** character-flavour Layer-1 machinery
  for any non-self-dual flavour ring.  SU(2)-flavour / unflavoured examples
  (`a1d*`, hexagon, pentagon, heptagon, …) are safe (self-dual).
* The convention `q = 𝖖` is correct; length-1 interactions are correct.
* The **BPS engine is the trusted oracle** here and gives the correct
  (axiom-satisfying) traces: `Tr(L_a)=δ_{a,1}+O(𝖖)`, `Tr(T_0^n)=O(𝖖^n)`.

## Fix directions (contract surface — HUMAN-DECIDE)

1. **Scope first**: strengthen the iso cert to weight-level and sweep all
   flavoured examples to map the blast radius before changing code.
2. **Weights not characters**: track flavour content on the Cartan weight
   lattice (fugacities), ρ = conjugation; characters emerge only after
   the trace.  Principled; touches the `RKAlgebra`/"free over R(SU(3))"
   design.
3. **Repair the relation tables**: keep characters but correct
   `_tt_dist2` (and any distance-≥2 with non-self-dual output) to encode
   the ⋆-resolved weight content; fix `_trace_reduce_word` to tag tile
   parity.  Localized but may not be expressible purely in characters.
4. **Route through BPS**: treat BPS as truth for SU3Ad, deprecate the
   standalone BPS-free `trace_layer1` + the (over-strong) iso claim, and
   fix the BPS flavour-charged-trace "trapezoid" separately.

## Reproduce

```bash
PYTHONPATH=. python -c "
import sys; sys.path.insert(0,'implementations')
from su3_ad_kalg import SU3ADKAlg
from laurent_poly import LaurentPoly
S=SU3ADKAlg(); B=S._bps_engine()
def transport(el):
    out={}
    for lab,lp in el.terms.items():
        bl=S._std_to_bps_label(lab); out[bl]=out.get(bl,LaurentPoly({}))+lp
    return {k:str(v) for k,v in out.items() if not v.is_zero()}
print('std :',transport(S.multiply((0,1,0,0,0),(4,1,0,0,0))))
print('bps :',{k:str(v) for k,v in B.multiply(S._std_to_bps_label((0,1,0,0,0)),S._std_to_bps_label((4,1,0,0,0))).terms.items()})
"
```

## RESOLUTION — `trace_layer1` confirmed correct, BPS-free (2026-06-13)

The standalone BPS-free Layer-1 reducer `trace_layer1` is **correct** once
two fixes are in place (both landed):

1. **`_tt_dist2` adjoint over-count** — fixed in #477 (`dd5ac1f`); see above.
2. **`Tr(D_odd) = ⋆Tr_D` tile parity** — `_trace_reduce_word` base case tags
   the D-letter index parity (`key = ('Tr_D', L[1] % 2)`); ρ inverts flavour
   fugacities, so odd-tile D generators carry the ⋆-dual character. Commit
   `a2f17da`.

**Validation (margin-safe).** Substituting the true elementary trace values
(`Tr_1, Tr_T, Tr_D` even, `⋆Tr_D` odd) into `trace_layer1(seed^a)` reproduces
the BPS trace **exactly**, for `seed ∈ {T_0, D_0}` and every power tested —
*within each power's margin-safe window*. A naive fixed-cutoff check shows a
spurious `T_0^3` mismatch at `q^7`: this is a **truncation artifact of the
verification, not a bug** — `trace_layer1(T_0^3)` carries coefficients down to
`q^{-8}`, so reproducing the trace at `q^k` needs the elementary values up to
`q^{k+8}`; with elementary values only to `q^{14}`, `q^7` is exactly the first
clipped order. With an adaptive margin (`K_safe = Kb + min(0, min_q_power)`)
the agreement is exact through the safe window for all powers. (`T_0^4` reaches
`q^{-15}`; this deep-negative-power structure is intrinsic, and is exactly why
naive finite-window Layer-2 attempts fail.)

So the Layer-1 pipeline is sound and BPS-free. The remaining open item is
**Layer 2** — the *values* `Tr_T`, `Tr_D` in closed form (BPS-free):

* `Tr_1` is closed-form (Kac–Wakimoto vacuum char of `\hat{sl}(3)_{-3/2}`,
  `sl3_m32_vacuum_char.py`, validated). ✓
* **Spectral-flow-of-vacuum is ruled out**: scanning `σ_λ̌(ch_0)` over a
  coweight grid (denominators 1–3) gives **0 hits** for both
  `G_T = Tr_T/(-𝖖)` and `G_D = Tr_D/(-𝖖)` (`sl3_scan.py`). Consistent with
  the manuscript: the admissible chars are dense per grade, so a finite
  *combination* (with cancellation), not a single flow, is required.
* **Pure-power orthonormality is (apparently) underdetermined**: the axiom
  `Tr(seed^a) = O(q^a)` is a *globally coupled* linear system (no order-by-
  order recursion — the `q^{-(a²-1)}` coefficient supports couple all orders).
  A constraint `(seed, a, k<a)` reaches unknowns up to index `≈ a²+a-2`, so at
  any cutoff `M` only powers with `a² ≲ M` are usable, and they cannot supply
  enough rows to pin the unknowns up to `M`. Richer constraints (mixed words
  with a known vanishing order, or constructively-built canonical basis
  elements) would be needed to close this route.
