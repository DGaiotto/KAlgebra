# Conjectures — Step 6 (the skein presentations)

The skein layer (`src/skein/`) realises the SU(2) skein algebra of a marked
surface `Σ` as the K-theoretic Coulomb branch algebra `A_𝖖[T[A₁, Σ]]`. It bears
directly on three structural statements, and — in the constructive spirit of the
project — it does not merely *check* them but *uses* the intrinsic (topological)
side as an independent oracle for the algebra it realises.

## 1. Orthonormality of the canonical basis

**Conjecture.** The skein algebras presented in `src/skein/` satisfy the
`K_𝖖`-algebra axioms; in particular, for the canonical basis `{L_a}` the Schur
pairing

    I_{a,b}(𝖖)  =  Tr( ρ(L_a) · L_b )   satisfies   I_{a,b}(𝖖) = δ_{a,b} + O(𝖖) :

the canonical basis is **orthonormal to leading order in `𝖖`**. (The `𝖖⁰` term
is the `Δ = spin = 0` identity sector — see `docs/conjectures-step1-samples.md`.)
For a bordered surface the pairing is read on the identity summand of the flavour
character; `ρ` rotates each irregular puncture by one marked point (regular
punctures are untouched, recovering `ρ² = 1` in the no-marked-point case).

## 2. The skein algebra is the Coulomb-branch algebra

**Conjecture (the `K_𝖖` dictionary).** For a marked surface `Σ`, the SU(2)
Kauffman-bracket skein algebra `Sk_{A₁}(Σ)` is the line-defect Schur sector
`A_𝖖[T[A₁, Σ]]` of the class-S theory `T[A₁, Σ]`, under the dictionary of the
step-6 note: simple closed curves ↦ Wilson-type character generators, open arcs
↦ monomial generators, Kauffman resolution ↦ the structure constants, and the
**quantum trace ↦ the "geometric F"** (the bare part of `F(−γ)`). The canonical
basis differs from the bare geometric curves by a triangular *dressing*; where a
closed form for that dressing is not in hand, the intrinsic side is used only as
an oracle and the realisation route fails honestly rather than guessing.

The evidence is **non-circular**: block A (the intrinsic Kauffman/stated engine)
knows nothing of the `KAlgebra` contract, so an agreement between a skein-side
computation and a BPS-side one is a genuine cross-check. Ingredient status
(each tagged [proven] / [measured] / [open] in the tier's own notes): the polygon
`[A₁,Aₙ]` identifications and the four-punctured-sphere = SU(2) `N_f=4` central
equality are [measured] on exact data; the intrinsic model of the line-defect
extension and the intrinsic (stated-side) half-index trace on general bordered
surfaces remain [open].

## 3. Flips are mutations; the vacuum anchors the realisation

An ideal triangulation of `Σ` is a BPS-quiver chart, and a **diagonal flip is a
quiver mutation**: `SkeinAtlas` presents one `Sk(Σ)` as an ensemble of charts
whose edge transitions are certified `KAlgebraIso`s (a bijective canonical-label
map intertwining `multiply`, `ρ`, and the trace). Composed flip loops realise the
rotation monodromy `ρ²` at the lifted level. Independently, the **vacuum Schur
index** computed through a skein chart equals the one computed from the
strong-coupling BPS quiver of the same theory (`bps_su2_nf1/2/3`), up to a
measured integral flavour-frame map — a check that shares no chart machinery
between the two sides.

## Verification scope

What the tests actually certify (`tests/test_skein_flows.py`):

| check | scope |
|---|---|
| intrinsic Kauffman algebra (contract-free) | `SkeinAlgebra` on the tetrahedron `S²`-with-4-punctures chart: a crossing resolves to 4 terms; `bar` is antimultiplicative on multicurve labels |
| two-parent architecture | a chart-ful instance runs the BPS routes; a chart-less instance raises a named honest-fail rather than guessing |
| roster + dispatchers | the named instances and the `polygon(n)` / `su2_nf(0..4)` dispatchers build and expose the expected generators |
| per-engine certification | pinned / native Kauffman (exact coefficients) / bordered chart / flavoured Clebsch–Gordan products agree with the stated engine |
| isos + object legs | per-instance `build_iso()` batteries; the `'skein-cone'` legs on `sqed1_object` / `u1hexagon_object` with coherence |
| contract axioms | bar, `ρ²`-twisted trace cyclicity, and orthonormality `I_{a,b}=δ+O(𝖖)` through the universal surface |
| `SkeinAtlas` | the closed GNO tower `L_γ²=L_{2γ}+L_γ+1`; a certified bordered flip; a gauged and a flavoured chart |
| vacuum anchor | the SU(2)+`N_f` skein-chart vacuum Schur index vs the strong-coupling BPS quiver, under the measured flavour-frame map |
