# The K_𝖖 conjecture for the skein algebra of the punctured sphere

A precise formulation (research task handed over 2026-06-10; n=4 is
the focus).  Every ingredient below is tagged **[proven]** (classical
mathematics), **[measured]** (verified exactly in this repo, n=4
chart unless stated), or **[open]** (the conjectural content).
Expert calibrations incorporated: the space of twisted traces is
finite-dimensional (so the O(𝖖) constraints are essential, not
redundant); ρ² = 1 for the regular punctured sphere; the RGKAlgebra
presentation *provides* a trace, with triangulation/mutation
independence expected but open.

---

## The object: which "version" of the skein algebra

Let C = S²₀,ₙ with n regular punctures, and Sk = Sk_{A₁}(C) the
Kauffman-bracket skein algebra over Z[𝖖^{±1/2}].

**Definition (the extension Sk̃ — the line-defect skein algebra).**
[open: intrinsic definition; measured: chart-level existence]
Sk̃ ⊃ Sk ⊗_{Z} Z[μ_p^{±1}] is generated additionally by the
**half-lattice sectors**: on any FST chart, the charge lattice of
Sk̃ is

    Λ̃  =  Λ_curves  +  Z·{f_p}  +  Z·(½ Σ_p f_p),

and more precisely all per-triangle parity classes occur
[measured: at n=4 the kernel lattice is the full Spin(8) weight
lattice; the canonical element at a curve label populates the classes
adj 11 / v 5 / s 5 / c 5 — triality-symmetric].  Candidate intrinsic
model [open]: a Roger–Yang-type algebra (arcs terminating at
punctures carry the half-unit flavour charges); the gold test for any
candidate is reproducing the measured spinor-sector dressings.

The conjecture is about **Sk̃**, not Sk: bare multicurves provably
fail orthonormality [measured: I(T,1) = 1 + (9-term flavour
character) + O(𝖖²)], and canonical elements have spinor-sector
content that no element of Sk ⊗ Z[μ^±] carries.

## The flavoured coefficient ring

**R = R(SU(2))^{⊗ n}** (one factor per regular puncture), as a
Z₊-ring with basis the irreducible characters χ_{j_1} ⊗ … ⊗ χ_{j_n}.
[measured at the Cartan level: flavour rank = n for all charts
tested, n = 4,5,6,7,8; the per-puncture SU(2) enhancement is
*guaranteed chart-by-chart* by the self-folded-triangle criterion —
a Z₂ quiver symmetry with charge difference ±f_p, verified for all
four punctures at n=4.]  At n=4 the four SU(2)'s assemble into
Spin(8) with triality permuting the v/s/c sectors
[measured: Tr(1) = 1 + χ₂₈ 𝖖² + O(𝖖³), the 28 branched as
⊕_p adj(SU(2)_p) ⊕ (2,2,2,2)].

The R-module structure on Sk̃: peripheral loops are the fundamental
characters [proven, classical]; `_label_section_decompose` folds the
flavour-lattice directions onto R-coefficients, with the half-lattice
class carrying the spinorial (half-odd-integer joint) characters.

## Bar involution

bar = the mirror map (reverse crossings, 𝖖 ↦ 𝖖⁻¹), antimultiplicative,
fixing every multicurve [proven].  Extension to Sk̃ fixing the
canonical basis [open, expected via the chart bar].

## ρ, with ρ² = 1

For the regular punctured sphere, **ρ² = 1** (expert input).
Measured ρ-data on the n=4 chart: curve labels are ρ-FIXED
(ρ(γ) = γ for all three generators and their multiples); ρ acts on
flavour monomials by ⋆ (μ ↦ μ⁻¹, i.e. the rep-ring duality), and
permutes half-lattice labels nontrivially.  Conjectural intrinsic
statement [open]: ρ|_{Sk} = id, ρ = ⋆ on the flavour lattice,
ρ = (an explicit involution) on the spinor sectors — so the twisted
trace below is honestly cyclic on Sk itself and twisted only on the
extension.

## The canonical basis

**[open: intrinsic existence/uniqueness; measured: chamber-level
existence + properties]**  A Z[𝖖^{±}]-basis {L_a} of Sk̃, bar-invariant,
containing 1, indexed by Λ̃-labels, such that:

* on each curve's tower it fuses as the **GNO-dual SO(3) irrep ring**:
  L_γ L_{kγ} = L_{(k+1)γ} + L_{kγ} + L_{(k−1)γ} with coefficients
  exactly 1 [measured, k ≤ 2 at n=4 AND n=5];
* the skein content of L_γ is (bare curve) + (dressing): even sector
  exactly 𝖖-unit·(Tr^lq(γ) + 1), the rest spinor-sector "bubbling"
  with [2]-type coefficients [measured];
* candidate chamber-free characterization [open]: the unique
  bar-invariant basis with positive fusion towers and the
  orthonormality below.

## The trace — the weak point, structured

**Existence [settled by construction]:** the RGKAlgebra presentation
provides a trace: pick any FST chart + finite chamber; Tr = the
Nahm-sum/measure construction.  The repo now supplies certified
chambers for every n (gauging recursion) and at n=4,5 multiple
independent chambers.

**Finite-dimensional ambient space [expert input]:** the space of
ρ²-twisted traces (ρ²=1 here: honest traces, i.e. functionals on
Sk̃/[Sk̃, Sk̃]) is finite-dimensional.  Hence cyclicity alone cannot
pin Tr; the **O(𝖖) constraints are essential**:

    (N1)  Tr(1) = 1 + O(𝖖)         (normalization)
    (N2)  Tr(L_a) = O(𝖖)  for a ≠ 0   (orthonormality against 1)
    (N3)  I_{a,b} = Tr(ρ(L_a) L_b) = δ_{a,b} + O(𝖖).

**Conjecture (trace uniqueness, n=4 first).**  On
Sk̃(S²₀,₄) ≅ (a variant of) the spherical DAHA(C∨C₁), the conditions
(N1)–(N3) select a unique element of the finite-dimensional trace
space — equivalently, the canonical-basis + trace pair is rigid.
The C∨C₁ spherical DAHA is the designated testing ground.

**MEASURED (2026-06-10,
`tests/verify_trace_space_dim.py`):**  at fixed generic parameters
(peripherals ↦ generic scalars, A generic) the trace space of
Sk(S²₀,₄) has

    dim = 5      (stable across windows kin = 2, 3, 4; upper bound,
                  exact if generator-cyclicity generates all
                  constraints in the window)

5 = b₂ of the smooth affine cubic surface (the Fricke character
variety of S²₀,₄; e = 6 = 1 + 5) — the Poisson-trace / HP₀ = H²
pattern for quantized symplectic surfaces (literature identification
to be verified).  So cyclicity leaves a 5-dim space and (N1)–(N3)
must cut 4 more dimensions + normalization: the counting is exactly
consistent with the uniqueness conjecture.

**Conjecture (invariance).**  The chamber trace is independent of
(i) the chamber (mutation/wall-crossing invariance) and (ii) the
triangulation.  Physically expected; not obvious.

**First instance MEASURED (2026-06-10,
`tests/verify_route_d_n4.py`):** the tetrahedral chart mutated at
(0,1) — a different cluster seed, with an independently auto-found
12-state chamber — gives Tr(1) and I(F_a,1) EXACTLY equal to the
tetrahedral values through 𝖖² after the canonical charge-basis
conversion.  Still open: the n=5 two-chamber comparison (bipyramid
20-spec vs trinion-glued 27-spec; both Tr(1) runs in flight), and
deeper 𝖖-orders.

**Identification (route B, n=4) — MEASURED 2026-06-10, exact through
𝖖².**  The trace = the Schur contour functional (SU(2) Schur measure
with four fundamental hypers), and the canonical curve tower maps to
the **bare gauge characters**: with the contour functional ⟨·⟩ of
`skein_sphere/schur_measure.py` (exact integer arithmetic) and the
triality frame map W_ch (hypers = puncture-doublet pairs,
x₁ = (e_p+e_q)/2 …, pairing = the curve's channel),

    Tr(1)        = ⟨1⟩        (so(8)_{-2} vacuum char; 𝖖 = q^{1/2})
    Tr(L_γ)      = ⟨χ₁(z)⟩    exactly at 𝖖² (dim-34 char), all THREE
                              channels, each in its own triality
                              frame — S-duality covariance measured
    I(L_γ, L_γ)  = ⟨χ₁·χ₁⟩    exactly at 𝖖² (dim-62 char) — the
                              quadratic/Gram test
    Tr(L_{2γ})   = ⟨χ₂⟩       0 through 𝖖³ ✓ (chamber K=2:
                              I(F_{2a},1) = 0 + O(𝖖³)); first nonzero
                              predicted the dim-259 char at 𝖖⁴
    Tr(L_γ)|𝖖³   = 0 ✓        (chamber K=3) — the odd-𝖖-power
                              selection rule of the measure holds

(`skein_sphere/tests/verify_schur_measure_n4.py`, 7 checks.)  No
dressing appears through 𝖖²: the naive insertion dictionary
L_{kγ} ↦ χ_k(z) holds so far, despite the constant (1,1,1) fusion —
the AW-vs-canonical discrepancy must show up, if at all, at 𝖖⁴
(K=3 now confirms 𝖖³ = 0) or in the 𝖖-corrections to orthogonality.
Caution from this computation: the per-puncture frame e_p and the
orthogonal frame x_i differ by the NON-Weyl triality map W_ch; naive
weight comparison without W produces a spurious "spinor bubbling
defect".  The fundamental-Wilson probe ⟨χ_{1/2}⟩ = q^{1/2}·8_v has
W-image outside the regular flavour lattice — the v-class extension
sector of Sk̃, as predicted by the line-defect-lattice picture.

## The conjecture, assembled (n = 4)

**There exists an extension Sk̃ of Sk(S²₀,₄) (conjecturally of
Roger–Yang type), an SU(2)⁴-flavoured coefficient structure
assembling to Spin(8), the mirror bar involution, an involution ρ
with ρ² = 1 acting trivially on curves and by ⋆ on flavour, a
bar-invariant canonical basis {L_a} with GNO-dual fusion towers, and
a trace — provided by any FST chamber and independent of all choices
— such that (Sk̃, bar, {L_a}, ρ, Tr) is an SU(2)⁴-flavoured
K_𝖖-algebra.  The trace is the unique twisted trace satisfying the
O(𝖖) constraints (N1)–(N3), and equals the Schur/Askey–Wilson
contour functional.**

Every clause has a measured instance behind it; the open mathematics
is: the intrinsic definition of Sk̃, intrinsic ρ on the spinor
sectors, existence/uniqueness of the canonical basis without a
chamber, the trace-space dimension + the (N1)–(N3) cut, and the
invariance statements.

## A structural conclusion (2026-06-10)

The canonical basis is **NOT** the Askey–Wilson-orthogonal family of
the measure: the exact measure-orthogonalization of the tower would
subtract ⟨χ₁⟩/⟨1⟩·1 = O(𝖖²)·(flavour characters) from χ₁, but the
chamber canonical maps to the BARE character χ₁ with no correction.
The right statement is:  **trace = the AW-type contour functional
(8-parameter symmetric locus t = q^{1/2}μ^{±1}); canonical basis = the
standard character basis**, with orthonormality holding in the
sharpened asymptotic form I(L_j, L_k) = δ + O(𝖖^{2|j−k|}) rather than
exactly.  The "AW-parameter dictionary" work item resolves this way.

## Work plan (n=4) — status 2026-06-10

1. **Trace space of spherical DAHA(C∨C₁)** [route A', corrected]:
   PARTIAL — dim = 5 measured on Sk(S²₀,₄) at generic fixed
   parameters (`tests/verify_trace_space_dim.py`); 5 = b₂(affine
   cubic).  Remaining: locate the chamber trace in the 5-dim space;
   show (N1)–(N3) cut it out (needs the symbolic-𝖖 version);
   literature cross-check (HP₀ of cubic surfaces, sDAHA traces).
2. **Measure-side pairings** [route B]: DONE at n=4 through 𝖖³ —
   exact equality, three channels, Gram test, selection rules
   (`schur_measure.py`, `tests/verify_schur_measure_n4.py`).  Open:
   𝖖⁴ (the 567/259 characters), exact-vs-asymptotic orthonormality.
3. **Chamber-independence** [route D]: n=4 mutated-chamber comparison
   DONE exactly (`tests/verify_route_d_n4.py`); n=5
   bipyramid-vs-glued in flight, with the measure-side prediction
   1 + 15𝖖² + 32𝖖³ + 132𝖖⁴ (odd power present at n=5!) from the
   general-n engine (`sphere_measure`).
4. **q⁰-trace closed form** [route E]: subsumed — Tr on the canonical
   tower is ⟨χ_k⟩ = O(𝖖^{2k}) with computable exact characters; the
   old q⁰ 9-term character of the BARE curve = the canonical-vs-bare
   change of basis at q⁰.
5. **Sk̃ candidate**: build the Roger–Yang-type extension on the
   chart dictionary and test its sectors against the measured
   v/s/c-class content (the W-image of the Wilson probe ⟨χ_{1/2}⟩
   pins the v-class sector normalization).
