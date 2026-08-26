"""`GNAbeKAlgebra` — `A_𝖖[(G, N)]` **natively on the `AbeKAlgebra` tier**:
gauge theory at any `RootDatum` with `T^*N` matter, presented on the enriched
rational quantum torus with no RG flow in the build path.

This is the top of the ladder the `(G, N)` work climbed (the design notes):

    GMatterOverPure        the RG flow (G,N) → (G,0), `S_RG = Ψ`; produced
      │                    `RG(a)` as an abelianized difference operator — the
      │                    DATA the rest was read off
      ↓
    matter_star_bubbling   the constraint calculus read off that data:
      │                    `F_{m'} = Z_{m'}·Q_{m'}`, `deg_μ Q = Δ_N(m) − Δ_N(m')`,
      │                    `Q[μ⁰] = pure`, solved per μ-sector by (★)+bar+`O(𝖖)`
      ↓
    matter_multislot       the same with a genuine multi-index `k⃗` (one μ-slot
      │                    per irreducible summand `N_i`)
      ↓
    GNAbeKAlgebra          THIS CLASS — the constructor packaged as a KAlgebra,
                           presented as `Q` (see `chart` below)

The 4d gauge group data
-----------------------
`lines` (a `global_form.LineLattice`) is the **4d gauge group** — a *maximal set
of mutually compatible Kapustin `(m, e)` labels* (user, 2026-07-28), the same
convention `PureGAbeKAlgebra` takes and the one that landed alongside the
`L_{m,e}` solver.  It is deliberately **not** "which cocharacter lattice": that
reading is the 3d one and is carried by `root_data.GaugeDatum` (the
discussion block in `global_form.py` spells out the difference).  The default is
the simply connected form, which is what the repo's coordinates realise
(cocharacters in `Q^∨`, weights in `P`, so the Dirac pairing is integral and the
lines are mutually compatible by construction).

The same lattice is handed to the inner pure-`G` algebra, so there is **one**
admission gate rather than two, and a non-simply-connected form honest-fails at
the same boundary `PureGAbeKAlgebra` documents: representable as a label
predicate, presented by a non-atom construction instead
(the design notes) — not on a fractional torus.

**No per-theory subclasses** (user ruling, 2026-07-28: *"use special names only
if you have algorithms specifically optimized for a G and/or N"*).  This one
class takes the group and the matter as arguments; a named class is earned by an
**optimized algorithm**, which today means the type-A `UNNfKAlgebra` /
`PureUNKAlgebra` and the native `PureSU2KAlgebra` — reachable from here via
`faster_equivalent()`.  Named theory *presets* (parameter tuples, not classes)
live in `g_matter_roster.ROSTER`.

Labels and ring
---------------
Labels are `((m, e), w)`.  `(m, e)` is the pure-`G` Kapustin 't Hooft–Wilson
charge in the tier's convention (`m` cochar-dominant, `e` Levi-dominant; `fold`
transports an arbitrary label there), and `w` is a **flavour irrep**.

**Flavour is `∏_i U(n_i)`** (user ruling, 2026-07-28: *"you could have `U(n_i)`
if there are `n_i` copies of the same irrep"*).  Write the matter as
`N = ⊕_i N_i^{⊕n_i}` with the `N_i` distinct; the slots carrying the same irrep
are interchangeable, so they carry a `U(n_i)`, not `n_i` separate `U(1)`s.  The
coefficient ring is therefore `⊗_i R(U(n_i))` (`zplus_ring.UNZPlusRing`, tensored
by `TensorZPlusRing` when there is more than one distinct irrep), and the
flavour-charged canonicals are the character multiples `L_{(g,w)} =
χ_w(μ)·L_{(g,1)}` — exactly as `UNNfKAlgebra` carries `χ_w` in a label slot.
This **supersedes** the earlier interim `AbelianZPlusRing(M) = R(U(1)^M)`
convention (user, 2026-07-27: *"just use a `U(1)` flavour for each for now"*).

Consequences worth stating, because they are visible:

* the label set is **no longer** the flow's.  `GMatterOverPure` still carries
  `k⃗ ∈ Z^M` (its μ-level grading is the RG cone, a different object), so the two
  agree on the flavour-NEUTRAL labels — which is what every flow comparison uses
  — but a charged label needs the weight-diagram translation below;
* structure constants get **shorter**: at SU(2)+2×2, `L_{(0,1)}·L_{(1,0)}` used
  to return four labels, two of them the separate `U(1)²` charges `(0,1)` and
  `(1,0)`; it now returns three, with those two packaged as the single `U(2)`
  doublet.  Nothing was lost — the dimensions match.

Relation to the standing D5 ruling: for a **U(N) gauge node** with `N_f`
fundamentals the flavour ring is `R(SU(N_f))`, because the diagonal
`U(1) ⊂ U(N_f)` lies in the gauge centre and is level bookkeeping.  A general `G`
need not have a centre to absorb it (`G₂` is centreless), so `R(U(n_i))` is the
right ring here and D5 stands where it was made.

The contract triple
-------------------
  * `torus_shape()` — one gauge node whose matter is declared as
    **representations** (`TorusShape` was widened for this, 2026-07-28: `Sp(4)+4`,
    `Spin(5)+5` and `Spin(5)+4ˢ` used to declare the same `1` and the tier could
    not say which theory it was).  The substrate it selects is
    `MatterWRQTorus`, whose rung ladder is now weight-indexed and is literally
    `matter_star_bubbling.matter_monomials`.
  * `chart(label)` — the **quotient** `Q` of the constructor's `F = Z·Q`,
    shifted by the label's flavour charge.  This is the one place where the
    constructor's output is not the chart: `F` is the *RG image*, dressed by the
    matter factor `Z`, and the substrate carries that dressing in its own
    cocycle `W = T_{−m'}(Z_m)·T_m(Z_{m'})/Z_{m+m'}`, so presenting `F` would
    count it twice.  Measured against the certified type-A oracle: `Q ==
    UNNfKAlgebra.chart` at U(2) `m = (0,−1),(0,−2),(−1,−1)` for `N_f = 1` and
    `N_f = 2` alike, where `F` is a full 3- or 9-sector μ-tower and the oracle's
    chart is a single level.
  * `decompose(x)` — level-ascending: at the lowest μ-level the slice is the
    **pure** chart of some canonical (that is ruling D3, `Q[μ⁰] = pure`,
    measured on every cell), so it is read by the pure-`G` WRQ engine with no
    target; subtract the level-shifted tower; recurse.  Honest-fails off scope.

Everything else — `multiply`, `ρ`/`ρ⁻¹`, `trace`, `inner_product`, W1 and the
`certify_canonical` acceptance — is derived by the tier.

Which algorithm actually builds `L_{m,e}` (asked, 2026-07-28)
------------------------------------------------------------
**Two, one per sector**, and they cost very differently — which is why the
benchmark (a probe in the source repository) times them separately.

*The `μ⁰` (pure) sector — constructive first, guarded solve last.*
`matter_multislot.solve_canonical_matter_vec` opens by asking
`PureGAbeKAlgebra(datum).chart((m, e))`, which dispatches routes in decreasing
preference, **each individually guarded (W1 + (★))** and falling through on
failure:

    wilson → minuscule → monoid[…] → closed_form → twist[k] → cone
           → peel[…] → star

    wilson       `m = 0`: the full-group character `χ_e`
    minuscule    `m` minuscule ⇒ the leading orbit IS the canonical
    monoid[h+r]  `e = 0`: the monoid law `L_{m,0}·L_{m',0} = L_{m+m',0}` is
                 EXACT, so an undressed monopole is a monomial in the cone
                 generators — no peel, no bar correction
    closed_form  top orbit `f = 1` + the bubbling cells in closed form
    twist[k]     the `e ↦ e + k·m̄` symmetry carries a built canonical to a
                 dressed one at no solve cost
    cone         `wrq_torus.build_canonical`
    peel[a·b]    product-and-peel from a lower seed + `kl_bar_correct`
    star         the LICENSED (★) solve (`star_bubbling.solve_canonical`)

`pure.route(label)` reports which one fired, and it is the most informative
number about a label.  MEASURED across the presets
(a probe in the source repository): identity/Wilson always `wilson`; the U(N)
monopoles `minuscule` (type A's luxury — ~1 ms); SU(2)/SU(3)/Sp(4)/Spin(5) and the
SU(2) adjoint `closed_form` (2–16 ms); and **`star` fires only at `G₂`** (9.3 s).
So the licensed solve is the *exception*, not the rule, on the pure side.

*The matter μ-tower — one constructive route, then the licensed (★)-guarded
solve.*

    twist[k=…]  →  solve

`twist[k]` is the **first constructive route on the matter side** (2026-07-28,
user-confirmed): `T^k : f_p ↦ v^{k·p̄}·f_p` carries `L^N_{m,0}` to
`L^N_{m, e+k·m̄}` in the matter theory exactly as it provably does in pure gauge,
so a dressed label on a twist cone costs **no solve** — build the undressed
canonical once and twist it.  Guarded like every other route: accepted only if
W1 holds and `well_formed()` names the intended label, otherwise it falls
through.  Measured **59×–1235× faster than the solve, on identical output**, with
the speedup *growing with charge depth* (SU(2)+1×2: 59× at `m=(1,)`, 269× at
`(2,)`, 1235× at `(3,)`; Sp(4)+1×4: 749×) — the twist is O(cells) while the solve
is superlinear in bubbled cells.  Scope: needs `m` in the **coroot span**, since
`m̄ = cochar_to_weight(m)` honest-fails on a central-torus direction (so U(N)'s
central directions have no twist cone — they are minuscule and already cheap).
`route(label)` reports which fired.

Everything else on the matter side is still solved, sector by sector.  `solve_canonical_matter_vec` walks the
μ-sectors `k⃗ ≤ Δ⃗_N(m)` in total-degree order (a linear extension of the
componentwise order), solving each with `solve_level` under the `O(𝖖)` offset
`Σ_{0⃗ ≠ i⃗ ≤ k⃗} Z[i⃗]·Q[k⃗−i⃗]`, subject to (★) + bar + `O(𝖖)` + the matter-side
μ-divisibility `F = Z·Q`.  The degree law (D2) is a **budget that PROPOSES**;
the **axioms DISPOSE** — a sector the budget calls forced is re-checked against
(★)+W1 (`verify_forced_level`) and falls through to the full solve if it fails,
which is what removed the circularity (TM6b).  Then `divide_by_Z_vec` extracts
`Q`, and the chart is `Q`, not `F`.

So the honest summary: *pure* is constructive wherever the routes reach — and
measured, it reaches everywhere except `G₂` — while *matter* is always the guarded
solve.  The benchmark makes the consequence sharp: at `G₂+1×7` the pure sector
costs 9.3 s and the matter tower then exceeds a 60 s budget (the full build was
timed at ~145 s), so **the expense is overwhelmingly the matter tower, not the
(★) solve**.  Within a fixed `G` the matter cost tracks the μ-sector count almost
linearly (Sp(4): 2 sectors 0.148 s → 4 sectors 0.366 s; Spin(5)+spinor: 0.154 s →
0.359 s), while rank and `|W|` predict nothing (U(3), rank 3 `|W|=6`, is 0.008 s;
Sp(4), rank 2 `|W|=8`, is 0.148 s) — the design record's claim, re-measured.

That asymmetry is the standing opportunity —
closed-form dressed generators on the matter side would be the constructive route
this tier still lacks (the design record, TM3's open item), and this class is the
**reference to beat**: a universal optimization (any `G`, any `N`) has to improve
the benchmark table across `(G, N)`, not at one point, and a *special* class is
warranted only where an algorithm is specifically optimized for a given `G`
and/or `N` (user ruling — those are `UNNfKAlgebra`, `PureUNKAlgebra`,
`UNQuiverKAlgebra`, `PureSU2KAlgebra`, `PureSUNKAlgebra`, reachable via
`faster_equivalent()`).

`multiply` — what was measured, and the bar law that governs it
--------------------------------------------------------------
`multiply = decompose(chart(a)·chart(b))` is the operation that exercises the
substrate cocycle **and** the level-ascending read together, so it is the
sharpest test of the presentation.  Closure is asserted by **exact
reconstruction** — `Σ_c C^c_{ab}·chart(c) == chart(a)·chart(b)` — never by
"decompose returned something", since a decomposition that lands off the span
would still return terms.  Measured: reconstruction holds at every pair tried
across SU(2)+2×2, U(2)+2×2, Sp(4)+2×4 and Spin(5)+2×4ˢ; associativity and the
unit law hold; and the **structure constants** (not merely the charts) equal
`UNNfKAlgebra`'s at U(2) for `N_f = 1` and `N_f = 2` alike.

The bar law on structure constants is the **conjugate-transpose** one.  Bar is
antimultiplicative and fixes the canonical basis, so `L_a L_b = Σ_c C^c_{ab}L_c`
gives `bar(L_aL_b) = L_bL_a` and hence

    bar(C^c_{ab}) = C^c_{ba}          — NOT   bar(C^c_{ab}) = C^c_{ab}.

An individual structure constant is 𝖖-palindromic exactly when the pair
**commutes**; these algebras are quantized, so that is the exception.  Measured:
`all-palindromic ⟺ commuting` in every case, with the Wilson × monopole pairs
precisely the non-commuting ones — at SU(2)+2×2, `C^{((1,),(1,))}` is `𝖖⁻¹` one
way and `𝖖` the other.  Do not "fix" that; it is the axiom working.

**Positivity and integrality are the sharper checks** (user, 2026-07-28).
Reconstruction proves the product lies in the *span*; positivity proves the
basis is *canonical* — `C^c_{ab} ∈ Z₊[𝖖,𝖖⁻¹]` is the Z₊-ring property
`zplus_ring` is named for, and a negative coefficient is the documented
signature of a fabricated element (the design notes), caught
where the bar-blind `𝖖⁰` self-norm was not.  Integrality: the scalar ring is
`Z[𝖖,𝖖⁻¹]` and nothing else (no `𝖖^{1/2}`, standing ruling).  Measured over
every ordered pair at SU(2)+2×2 (incl. flavour-charged labels), U(2)+N_f=1,2,
Sp(4)+2×4 and Spin(5)+2×4ˢ, with `UNNfKAlgebra` as a control: 300+ structure
constants, **no negative and no non-integer coefficient**.  Note the scope —
positivity is a property of structure constants, *not* of the Schur index,
which legitimately carries negatives (`1 − 5𝖖²`).

Discipline
----------
The tier still exposes **no solve entry point**: `chart` calls into
`matter_star_bubbling`, which applies the whole guard — (★) affine-Weyl residue
cancellation, W1 bar, the `O(𝖖)` bubbling condition, and the matter-side
μ-divisibility `F = Z·Q` with its degree law — before returning anything, and
raises `DivisibilityFailure` rather than fabricating.  That is the (★)-guarded
solve licensed by the user ruling of 2026-07-27, with matter's fourth condition
on top: (★) does **at least as much work here as in `(G,0)`, and strictly
more** — measured, affine-Weyl partner pairs with unequal `Δ_N` outnumber the
equal ones (49 vs 100 in the census), and those couple the μ-levels, which is
work (★) does not do in the pure theory at all.

Scope, stated rather than papered over
--------------------------------------
* Verified against `GMatterOverPure` at 16 labels across SU(2), SU(3), Sp(4),
  Spin(5) (vector **and** spinor matter), SU(4), Sp(6) and G₂, with 0
  mismatches; independently cross-checked in the type-A corner by handing built
  elements to `UNNfKAlgebra.decompose`, which recovers the documented
  Gaussian-binomial spread and the `N_f ≥ 2` bubbling channel verbatim.
* What bounds the reach is **cost**, not correctness, and cost tracks the number
  of bubbled cells carrying free parameters under the degree law — not rank, not
  `|W|`, not exceptionality (Sp(6) at rank 3 with `|W| = 48` costs 14 s; Sp(4) at
  rank 2 with `m = (2,1)` costs 900 s).
* Everything `PureGAbeKAlgebra` honest-fails on, this honest-fails on — but odd
  `⟨Σ⁺, m⟩` cocharacters are **no longer among them** (ruling D31 retracts the
  "theorem, not a gap" this list used to cite; they are ordinary atoms now, and
  the standard-`Z²` BPS chart at SU(2)/SO(3) is an independent presentation
  rather than the only one that reaches them).

Run `g_matter_abe_kalgebra` in the source repository for a smoke
tour; certification is the suite in the source repository.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from abe_kalgebra import AbeKAlgebra, TorusShape
from kalgebra import Element, Label
from laurent_poly import LaurentPoly
from root_datum import RootDatum
from wrq_torus import (decompose as _wrq_decompose,
                       rho_label as _wrq_rho_label,
                       rho_level_star as _wrq_rho_level_star)
from zplus_ring import RElement, RPowerSeries, TensorZPlusRing, UNZPlusRing

from pure_g_abe_kalgebra import (PureGAbeKAlgebra,
                                 _normalize_optimizations)


__all__ = ["GNAbeKAlgebra"]


class GNAbeKAlgebra(AbeKAlgebra):
    """`(G, N)` on the abelianized contract — one class for any `RootDatum` and
    any matter representation.

    ``GNAbeKAlgebra(root_datum.sp_n(2), (1, 0), nf=2)`` is `Sp(4)` with two
    fundamentals; ``GNAbeKAlgebra(root_datum.b_n_simply_connected(2),
    [((0, 1), 2)])`` is `Spin(5)` with two **spinor** hypers; the `matter`
    argument takes the same declarations as `GMatterOverPure` (a bare highest
    weight, a list of weights, or `(weight, multiplicity)` pairs)."""

    #: The declared optimizations, and **which sector of this theory each one
    #: governs**.  The register is shared with `PureGAbeKAlgebra` so that one
    #: switch means one thing everywhere: `optimizations=()` turns everything off
    #: in both sectors, and a name switched on is on wherever it is sound.
    #:
    #: | name | governs | why there and not elsewhere |
    #: |---|---|---|
    #: | `theta_twist` | **both** the matter towers and the inner pure charts | it assumes nothing about `G` or `N` — the universal entry (user, 2026-08-25: *"the θ-twist applied to general `(m,e)` is a useful optimization, and other obsolete optimizations should be left maybe to specific specializations of `AbeKAlgebra` which make assumptions on `G` and `N`"*).  It survives matter as `matter_star_bubbling.matter_theta_twist` (ruling TM10) |
    #: | `monoid` | the **inner pure sector only**, never the matter towers | the monoid law's proof is the pure-gauge spectral form, and a free monoid cannot carry Littlewood–Richardson multiplicities, so `L^N_{m,0}·L^N_{m',0} = L^N_{m+m',0}` at `(G, Adj)` would be *"very suspicious and suggests something badly wrong"* (user ruling, 2026-07-30).  `self._pure` genuinely IS the `N = 0` theory, which is the specialization the law belongs to |
    #:
    #: So `monoid` is accepted here as a name — turning it off must be possible
    #: from the object the caller holds — while `_base_tower` never consults it.
    DEFAULT_OPTIMIZATIONS = PureGAbeKAlgebra.DEFAULT_OPTIMIZATIONS
    KNOWN_OPTIMIZATIONS = PureGAbeKAlgebra.KNOWN_OPTIMIZATIONS

    def __init__(self, datum: RootDatum, matter, nf: int | None = None,
                 allow_solve: bool = True, strict_guard: bool = False,
                 pad: int = 1, lines=None, pad_escalate: int = 2,
                 pad_retry_budget: float = 60.0,
                 constructive_routes: bool = False,
                 optimizations=None) -> None:
        from g_matter_over_pure import _normalize_matter
        self.datum = datum
        # Stage 1 (2026-08-25): the axioms are the whole production surface, so
        # the θ-twist shortcut is OFF by default here exactly as the pure tier's
        # constructive zoo is — the twist law is MEASURED, not proved, which is
        # precisely the kind of special case stage 1 removes and stage 2 may
        # reintroduce as a declared optimization.  `constructive_routes=True`
        # restores it (and the inner pure algebra's zoo with it).
        self.constructive_routes = bool(constructive_routes)
        # Stage 2 (user direction, 2026-08-25: "restore some of the
        # optimizations, leaving the option to turn them off").  The declared
        # optimizations of the matter tier are the SAME register as the pure
        # tier's — `PureGAbeKAlgebra.DEFAULT_OPTIMIZATIONS` — because the one
        # entry, the θ-twist, survives matter: `matter_star_bubbling.
        # matter_theta_twist` carries `L^N_{m,0}` to `L^N_{m,k·m̄}` exactly as the
        # pure `theta_twist` does, and it had to commute with the `Z`-division
        # and the μ-grading to do so (ruling TM10, measured 7/7).  The set is
        # handed to the inner pure algebra too, so one switch governs both
        # sectors rather than two that can disagree.
        self.optimizations = _normalize_optimizations(
            optimizations, self.DEFAULT_OPTIMIZATIONS,
            self.KNOWN_OPTIMIZATIONS, type(self).__name__)
        # The **4d gauge group data** — `lines`, a `global_form.LineLattice`:
        # a MAXIMAL set of mutually compatible Kapustin `(m, e)` labels (user,
        # 2026-07-28), which is the same convention `PureGAbeKAlgebra` takes and
        # which landed alongside the `L_{m,e}` solver.  It is NOT "which
        # cocharacter lattice" — that reading is the 3d one, carried by
        # `root_data.GaugeDatum`; see the discussion block in
        # `global_form.py`.  Default = the simply connected form, which is what
        # the repo's coordinates realise.  The inner pure-`G` algebra is handed
        # the same lattice, so a label is admitted here iff it is admitted there
        # — one gate, not two.
        if lines is None:
            from global_form import simply_connected_lines
            lines = simply_connected_lines(datum)
        self.lines = lines
        if nf is not None:
            matter = ((tuple(matter), int(nf)),)
        self._matter = _normalize_matter(datum, matter)
        if not self._matter:
            raise ValueError(
                "GNAbeKAlgebra: no matter — use PureGAbeKAlgebra directly")
        self._M = len(self._matter)
        self._pad = int(pad)
        #: how far to WIDEN the solve box before honest-failing.  An
        #: `inconsistent` verdict is the box's too-narrow signature
        #: (`solve_level`), so a failure at the default pad is a
        #: statement about the window, not the label — measured at
        #: Spin(5)+Adj, m=(1,0): raises at pad=1, builds at pad=2.
        self._pad_escalate = int(pad_escalate)
        #: seconds — do NOT widen the box after a failure that already
        #: cost more than this.  A retry re-runs the whole per-sector
        #: solve, so on an expensive label it multiplies the failure
        #: (G₂+Adj m=(1,0): ~1.7×/sector, ~105 min for one attempt),
        #: while where it pays the base attempt is cheap (Spin(5)+Adj
        #: m=(1,0): fails in 1.3 s, builds at pad=2 in 1.6 s).
        self._pad_retry_budget = float(pad_retry_budget)
        # --- the matter must be a representation OF THIS GLOBAL FORM ---------
        # A rep of `G̃/H` is a rep of `G̃` on which `H` acts trivially, i.e. all of
        # whose weights lie in the form's ELECTRIC lattice.  So `PSU(3) + 3` is
        # not a theory — the fundamental carries centre charge — while
        # `PSU(3) + 8` (the adjoint) is.  Checked here because the symptom
        # otherwise surfaces far away and unrecognizably: `Δ⃗_N(m)` involves
        # `⟨m, w⟩` over the matter weights, which goes fractional for a
        # centre-charged `w` against a fractional coweight, and the solver dies in
        # `range()` with a `Fraction` (measured, 2026-07-28).
        if self.lines.H:
            from wrq_torus import levi_character
            zero = (0,) * datum.dim
            for lam in self._matter:
                chi = levi_character(datum, zero, tuple(lam))
                bad = [tuple(w) for w in chi.terms
                       if not self.lines.elec_admits(tuple(w))]
                if bad:
                    raise ValueError(
                        f"GNAbeKAlgebra: the matter {tuple(lam)} is not a "
                        f"representation of {self.lines.name} — its weights "
                        f"{bad[:3]}{'…' if len(bad) > 3 else ''} carry centre "
                        f"charge, so they are outside this form's electric "
                        f"lattice.  A rep of G̃/H is one on which H acts "
                        f"trivially: at PSU(N) the fundamental is not a "
                        f"representation, the adjoint is.")
        # Flavour: `∏_i U(n_i)` for `n_i` copies of the SAME irrep (user ruling,
        # 2026-07-28).  Group the slots by distinct highest weight, keeping the
        # slot indices so the Cartan weights of `χ_{λ_i}` land on the right
        # μ-slots of the substrate (no reordering of slots).
        groups: list = []
        seen: dict = {}
        for idx, lam in enumerate(self._matter):
            if lam not in seen:
                seen[lam] = len(groups)
                groups.append((lam, []))
            groups[seen[lam]][1].append(idx)
        self._groups = tuple((lam, tuple(idxs)) for lam, idxs in groups)
        facs = [UNZPlusRing(len(idxs)) for _lam, idxs in self._groups]
        self._facs = tuple(facs)
        self._R = facs[0] if len(facs) == 1 else TensorZPlusRing(facs)
        # The inner algebra is genuinely pure gauge, so the whole set is handed
        # through unchanged — `monoid` included, and sound there.  One switch,
        # one meaning: `optimizations=()` here turns everything off in both
        # sectors.  What keeps the ruling is not a filter but `_base_tower`,
        # which consults `theta_twist` and nothing else.
        self._pure = PureGAbeKAlgebra(datum, allow_solve=allow_solve,
                                      strict_guard=strict_guard,
                                      lines=self.lines,
                                      constructive_routes=self.constructive_routes,
                                      optimizations=tuple(sorted(self.optimizations)))
        self._base: dict = {}          # {(m, e): MatterWRQTorus} — the towers
        self._route: dict = {}         # {(m, e): which route built it}

    def __repr__(self) -> str:
        reps = ",".join(str(lam) for lam in self._matter)
        return f"GNAbeKAlgebra({self.datum.name}; {reps})"

    # ----- what theory this is ------------------------------------------

    @property
    def M(self) -> int:
        """The number of hypermultiplet slots."""
        return self._M

    @property
    def groups(self) -> tuple:
        """`((highest weight λ_i, slot indices), …)` — the matter grouped by
        DISTINCT irrep.  `n_i = len(slot indices)` is the multiplicity, and the
        flavour group is `∏_i U(n_i)`."""
        return self._groups

    @property
    def matter(self) -> tuple:
        """The matter highest weights, one per hypermultiplet slot."""
        return self._matter

    def pure(self) -> PureGAbeKAlgebra:
        """The pure-`G` algebra whose canonical basis the `μ⁰` slices land in
        (D3) — the decompose engine, not a flow container."""
        return self._pure

    def torus_shape(self) -> TorusShape:
        """One gauge node carrying the matter **representations** — the widened
        shape, which is what lets this theory name itself."""
        return TorusShape.from_root_data((self.datum,), matter=(self._matter,))

    # ----- labels --------------------------------------------------------

    def coefficient_ring(self):
        return self._R

    def identity(self) -> Label:
        z = (0,) * self.datum.dim
        return ((z, z), self._R.one_basis())

    # ----- flavour: characters of `∏_i U(n_i)` ⇄ substrate μ-levels -------
    #
    # The substrate grades residuals by a μ-level `k⃗ ∈ Z^M`, one component per
    # hypermultiplet SLOT.  The coefficient ring is graded by irreps of
    # `∏_i U(n_i)`.  The two are related by the weight diagram: `χ_λ` expands
    # into its Cartan weights, and those weights ARE μ-levels once placed on the
    # group's slots.  These two helpers are that translation, in both
    # directions; everything flavour-side in this class goes through them.

    def _factor_bases(self, w):
        """A ring basis element → the per-group tuple of `U(n_i)` weights."""
        if len(self._facs) == 1:
            return (self._facs[0].reduce(w),)
        if len(w) != len(self._facs):
            raise ValueError(
                f"{self!r}: flavour label {w} has {len(w)} factors, expected "
                f"{len(self._facs)}")
        return tuple(R.reduce(b) for R, b in zip(self._facs, w))

    def _mk_flav(self, per_group):
        """The inverse of `_factor_bases`."""
        per_group = tuple(per_group)
        return per_group[0] if len(self._facs) == 1 else per_group

    def _slot_weight_sets(self) -> tuple:
        """One expanded weight set per hypermultiplet slot (the `slots` of
        `wrq_torus.rho_label` / `rho_level_star`)."""
        if not hasattr(self, "_slot_wts"):
            from matter_wrq_torus import rep_weights
            self._slot_wts = tuple(rep_weights(self.datum, lam)
                                   for lam in self._matter)
        return self._slot_wts

    def _rho_closed_form(self, a: Label, inverse: bool) -> Label:
        """The explicit label-level ρ^{±1} (promoted 2026-08-23; the twist
        route is the verifier `verify_rho_via_twist`): gauge part by
        `wrq_torus.rho_label` with this theory's matter slots; flavour irrep
        by `w_i ↦ w_i^⋆ ⊗ det_i^{−D_i}` — the ring's ⋆ (rep-ring duality,
        `star_basis`) and the level star's per-group rung count `D_i`
        (`rho_level_star`; slots of one group carry the same representation,
        so `D` is constant across them)."""
        (m, e), w = a
        slots = self._slot_weight_sets()
        g = _wrq_rho_label(self.datum, m, e, slots=slots, inverse=inverse)
        D = _wrq_rho_level_star(slots, m, inverse=inverse)
        per = []
        for R, lam, (_l, idxs) in zip(self._facs, self._factor_bases(w),
                                      self._groups):
            dual = R.star_basis(lam)
            Di = D[idxs[0]]
            per.append(R.reduce(tuple(x - Di for x in dual)))
        return (g, self._mk_flav(per))

    def rho(self, a: Label) -> Label:
        return self._rho_closed_form(a, inverse=False)

    def rho_inverse(self, a: Label) -> Label:
        return self._rho_closed_form(a, inverse=True)

    def _flav_levels(self, w) -> dict:
        """`{μ-level k⃗ ∈ Z^M: multiplicity}` — the weight diagram of the
        character `χ_w`, with each group's weights placed on that group's
        slots.  This is what makes `L_{(g,w)} = χ_w(μ)·L_{(g,0)}` computable."""
        acc = {(0,) * self._M: 1}
        for R, lam, (_l, idxs) in zip(self._facs, self._factor_bases(w),
                                      self._groups):
            nxt: dict = {}
            for wt, mult in R.character(lam).items():
                for base, c in acc.items():
                    k = list(base)
                    for j, i in enumerate(idxs):
                        k[i] += int(wt[j])
                    key = tuple(k)
                    nxt[key] = nxt.get(key, 0) + c * mult
            acc = nxt
        return {k: c for k, c in acc.items() if c}

    def _flav_unbranch(self, levels: dict) -> dict:
        """`{k⃗: int} → {ring basis: int}` — un-branch a μ-graded flavour
        content into `∏_i U(n_i)` characters, group by group (the groups own
        disjoint slot sets, so the characters are products and the
        decomposition factorizes).  Honest-fails via `sun_characters.decompose`
        on content that is not a genuine character combination."""
        def rec(gi, lev):
            if gi == len(self._groups):
                return {(): next(iter(lev.values()))} if lev else {}
            R = self._facs[gi]
            idxs = self._groups[gi][1]
            # bucket by the OTHER slots' levels, decompose in this group's
            buckets: dict = {}
            for k, c in lev.items():
                rest = tuple(k[i] for i in range(self._M) if i not in idxs)
                mine = tuple(k[i] for i in idxs)
                buckets.setdefault(rest, {})[mine] = \
                    buckets.setdefault(rest, {}).get(mine, 0) + c
            # decompose each bucket, then regroup by the resulting λ
            bylam: dict = {}
            for rest, poly in buckets.items():
                for lam, mult in R.from_abelian(poly).items():
                    bylam.setdefault(lam, {})[rest] = \
                        bylam.setdefault(lam, {}).get(rest, 0) + mult
            out: dict = {}
            for lam, restlev in bylam.items():
                # re-embed `rest` into full M-vectors for the recursion
                full = {}
                other = [i for i in range(self._M) if i not in idxs]
                for rest, c in restlev.items():
                    k = [0] * self._M
                    for j, i in enumerate(other):
                        k[i] = rest[j]
                    full[tuple(k)] = c
                for tail, c in rec(gi + 1, full).items():
                    out[(lam,) + tail] = out.get((lam,) + tail, 0) + c
            return out

        raw = rec(0, {k: int(c) for k, c in levels.items() if c})
        return {self._mk_flav(b): c for b, c in raw.items() if c}

    def flavour_levels(self, w) -> dict:
        """`{μ-level k⃗ ∈ Z^M: multiplicity}` — the weight diagram of `χ_w`,
        placed on the hypermultiplet slots.  The public face of the ring ⇄
        substrate translation this class runs on.

        Exposed because consumers outside this class genuinely need it: the
        substrate grades everything by μ-level while the coefficient ring is
        graded by irreps of `∏_i U(n_i)`, so anything that reads a chart and
        reports an `RElement` has to cross that seam (any consumer that reads a
        chart's flavour content does).  Without a public face the only route is the private
        helper, which is exactly the internals-tunnelling the design notes names
        as the recurrent failure mode."""
        return self._flav_levels(w)

    def flavour_from_levels(self, levels: dict) -> dict:
        """`{μ-level k⃗: int} → {ring basis element: int}` — the inverse
        direction, un-branching μ-graded content into `∏_i U(n_i)` characters.

        Honest-fails (via the ring's own `from_abelian`) on content that is not
        a genuine character combination, which is the useful behaviour: a
        consumer that has assembled flavour content wrongly finds out here
        rather than reporting a plausible `RElement`."""
        return self._flav_unbranch(levels)

    def _split(self, label):
        """`((m, e), w)` → the folded gauge label and the flavour irrep."""
        g, w = label
        return (self._pure.fold(tuple(g[0]), tuple(g[1])),
                self._mk_flav(self._factor_bases(w)))

    def fold(self, m, e, w=None) -> Label:
        """Transport an arbitrary `(m, e)` to the tier's frame (`m`
        cochar-dominant, `e` Levi-dominant), carrying the flavour irrep."""
        return (self._pure.fold(m, e),
                self._R.one_basis() if w is None
                else self._mk_flav(self._factor_bases(w)))

    def r_label_decompose(self, label):
        """Flavour-lift coordinate: the section is the flavour-neutral label,
        the `∏_i U(n_i)` basis key is the flavour irrep `w` (`χ_w`)."""
        g, w = self._split(label)
        return (g, self._R.one_basis()), w

    def r_label_compose(self, section, r_basis_label):
        return (section[0], self._mk_flav(self._factor_bases(r_basis_label)))

    def _label_section_decompose(self, label):
        g, w = self._split(label)
        return (g, self._R.one_basis()), RElement(self._R, {w: 1})

    def route(self, label) -> str:
        """Which route built the matter tower of `label` — `'twist[k=…]'` (free,
        the θ-twist) or `'solve'` (the licensed (★)-guarded solve).  Builds it if
        necessary, mirroring `PureGAbeKAlgebra.route`.

        The pure sector's own route is `pure().route((m, e))`, which is the
        richer dispatch (`wilson`/`minuscule`/`monoid`/`closed_form`/`twist`/
        `cone`/`peel`/`star`).  Two reporters because the two sectors are two
        different algorithms — see the module docstring."""
        g, _w = self._split(label)
        self._base_tower(g)
        return self._route[g]

    # ----- the 4d gauge group data --------------------------------------

    @property
    def theory(self) -> str:
        """Physics-facing description, derived from the group data and the
        matter rather than declared per subclass — there are no per-theory
        subclasses (user ruling, 2026-07-28: special names are earned by an
        *optimized algorithm*, not by being a nameable theory)."""
        reps = ", ".join(f"{len(idxs)}×{lam}" for lam, idxs in self._groups)
        return f"{self.lines.name} + {reps}"

    def flavour_group(self) -> str:
        """`∏_i U(n_i)` written out — the faithful flavour group under ruling
        TM7, grouping matter slots by identical highest weight."""
        parts = [f"U({len(idxs)})" for _lam, idxs in self._groups]
        return " × ".join(parts) if parts else "trivial"

    # ----- ties to the rest of the tier ---------------------------------

    def removal_flow(self, drop=None):
        """The `GMatterOverMatter` flow decoupling some hypers from this theory,
        landing on the `(G, N_keep)` algebra.  `drop` is a slot index or an
        iterable of them; the default drops the last slot."""
        from g_matter_over_matter import GMatterOverMatter
        if drop is None:
            drop = len(self._matter) - 1
        return GMatterOverMatter.from_uv(self.datum, self._matter, drop=drop)

    def removal_tower(self):
        """The one-slot-at-a-time removal tower down to pure `G` — this theory's
        place in the family, as a list of certified flows."""
        from g_matter_over_matter import matter_removal_tower
        return matter_removal_tower(self.datum, self._matter)

    def flow_twin(self):
        """The same abstract algebra presented as an **RG flow** over pure `G`
        (`GMatterOverPure`) instead of natively — the other presentation."""
        from g_matter_over_pure import GMatterOverPure
        return GMatterOverPure(self.datum, self._matter)

    def flow_iso(self):
        """A certified `KAlgebraIso` to the flow presentation where one exists,
        else `None` — the Goal-1.3 "one object, two presentations" certificate.

        **Exists exactly when all matter irreps are distinct** (every `n_i = 1`):
        the native ring `⊗_i R(U(n_i))` then coincides with the flow's
        `R(U(1)^M)`, the flavour Cartan restriction is the identity, and the two
        label sets are the *same tuples* — so it is
        `KAlgebraIso.identity_on_labels`, the strongest form of the certificate.
        When an irrep repeats there is only an **embedding** (the image is the
        flavour-symmetric part of the flow's finer grading), and `None` is
        returned rather than a map mislabelled as an iso.

        Trace caveat (measured): `verify_trace_equivariant` reports `False` at
        `M = 1` although the two traces are *identical as series*, because
        `UNZPlusRing(1)` and `AbelianZPlusRing(rank=1)` are isomorphic but not
        **equal** and `RPowerSeries.__eq__` compares the ring tag.  Use
        `verify_flow_trace`."""
        if any(len(idxs) > 1 for _lam, idxs in self._groups):
            return None
        from kalgebra_iso import KAlgebraIso
        return KAlgebraIso.identity_on_labels(
            self, self.flow_twin(),
            name=f"{self.theory}: native ≅ GMatterOverPure")

    def verify_flow_trace(self, a, K: int = 4) -> bool:
        """`Tr_native(L_a) == Tr_flow(L_a)` through the flavour Cartan hom, which
        sidesteps the isomorphic-but-unequal ring tags described in `flow_iso`.
        Wired at `M = 1`, where `un_to_cartan_hom(1)` is the whole dictionary."""
        if self._M != 1:
            raise NotImplementedError(
                f"{self!r}.verify_flow_trace: only M = 1 is wired (the "
                f"⊗_i R(U(1)) → R(U(1)^M) hom for M = {self._M} is not shipped)")
        from zplus_ring import un_to_cartan_hom
        h = un_to_cartan_hom(1)
        return h.apply_RPowerSeries(self.trace(a, K=K)) == \
            self.flow_twin().trace(a, K=K)

    def faster_equivalent(self):
        """A certified, materially faster presentation of the *same* algebra
        where one exists, else `None` — i.e. the places where a **specially
        named class earns its name by an optimized algorithm** (user ruling,
        2026-07-28) rather than by naming a theory.

        Today that is the type-A corner only: `UNNfKAlgebra(N, N_f)`, whose
        `chart` this class's was measured equal to (ruling TM6).  Its
        coefficient ring is `R(SU(N_f))` rather than `R(U(N_f))`, so it is the
        same algebra over a *specialized* flavour ring, not a drop-in — see
        `g_matter_un_nf_seam`.  `None` everywhere else, deliberately."""
        if not self.datum.name.startswith("U("):
            return None
        if len(self._groups) != 1:
            return None
        from matter_wrq_torus import defining_weight
        if self._groups[0][0] != tuple(defining_weight(self.datum)):
            return None
        from un_nf_kalgebra import UNNfKAlgebra
        return UNNfKAlgebra(self.datum.dim, self._M)

    # ----- the contract triple ------------------------------------------

    def _twist_source(self, m, e):
        """`(source gauge label, k)` with `T^k·L^N_source = L^N_{(m,e)}`, or
        `None` — the general θ-twist over the MATTER tier's own memo of towers.

        The same statement as `PureGAbeKAlgebra.twist_source`, and for the same
        reason (user, 2026-08-25: *"every known `(m,e)` means we also know its
        θ-twists, not just `m,0`"*): `matter_theta_twist` carries `L^N_{m,e}` to
        `L^N_{m,e+k·m̄}` at any `e`, so the electric labels at fixed `m` fall into
        `m̄`-lines and one known point gives the whole line, in both directions.

        That the twist survives matter at all is ruling TM10 and was not
        automatic: the matter chart is the quotient `Q` of `F = Z·Q` and the
        dressing `Z(m)` carries `v`-dependence, so `T^k` had to commute with the
        `Z`-division and with the μ-grading (measured 7/7).  What is *not*
        assumed here is that a matter tower's line matches the pure one's: the
        arithmetic of which pairs are `T`-related is a fact about `m̄` alone
        (`twist_shift`), which is why that rule is shared while the memo scanned
        is this tier's."""
        m, e = tuple(m), tuple(e)
        best = None
        for (mc, ec) in self._base:
            if mc != m or ec == e:
                continue
            k = self._pure.twist_shift(m, tuple(a - b for a, b in zip(e, ec)))
            if k is None:
                continue
            if best is None or abs(k) < abs(best[1]):
                best = ((m, ec), k)
        if best is not None:
            return best
        k0 = self._pure.twist_level(m, e)
        if k0:
            return ((m, (0,) * self.datum.dim), k0)
        return None

    def _base_tower(self, g):
        """`L_{(m,e),0⃗}` as a `MatterWRQTorus` — **the quotient `Q`**, cached.

        The constructor solves the RG image `F_{m'} = Z_{m'}·Q_{m'}`; the chart
        of the canonical is the un-dressed `Q`, because the matter dressing `Z`
        is carried by the substrate's own cocycle
        (`W = T_{−m'}(Z_m)·T_m(Z_{m'})/Z_{m+m'}`) and would otherwise be counted
        twice.  Measured against the certified type-A oracle: `Q ==
        UNNfKAlgebra.chart` at U(2) `m = (0,−1),(0,−2),(−1,−1)` for both
        `N_f = 1` and `N_f = 2` — while `F` is a whole μ-tower there and the
        oracle's chart is one level.  `Q[0⃗] = pure` (ruling D3) is what makes
        `decompose` below a read rather than a solve.

        Every guard lives inside `solve_canonical_matter_vec`; the division is
        re-checked here and honest-fails if the built `F` is not divisible."""
        x = self._base.get(g)
        if x is None:
            from matter_multislot import (divide_by_Z_vec,
                                          solve_canonical_matter_vec)
            from matter_star_bubbling import (DivisibilityFailure,
                                              matter_theta_twist)
            m, e = g
            # The 4d gauge group gate, in the same form PureGAbeKAlgebra uses.
            # `_pure.chart` would apply it, but the matter tower is solved
            # directly by `solve_canonical_matter_vec`, so it must be applied
            # here too or a line outside the global form would be built.
            if not self.lines.admits(m, e):
                raise NotImplementedError(
                    f"{self!r}: L_{g} is not a line of {self.lines.name} — the "
                    f"4d gauge group's maximal mutually-compatible (m, e) "
                    f"sublattice excludes it")
            if not self.lines.abe_representable(m):
                raise NotImplementedError(
                    f"{self!r}: L_{g} was refused by abe_representable.  Since "
                    f"ruling D31 that predicate is True for every charge on the "
                    f"default phase — odd ⟨Σ⁺,m⟩ included, since the honest "
                    f"half-integral atom phase is restored at the cocycle through "
                    f"its integral coboundary — so reaching this path means the "
                    f"datum carries an explicit phase_is_canonical= override.  "
                    f"See a probe in the source repository.")
            # --- the θ-TWIST route (2026-07-28) --------------------------
            # `T^k : f_p ↦ v^{k·p̄}·f_p` carries `L^N_{m,0}` to `L^N_{m,k·m̄}`,
            # measured 7/7 and user-confirmed, so a dressed label on a twist cone
            # costs NO SOLVE — the first constructive route on the matter side.
            # Guarded exactly like the pure routes: build the undressed
            # canonical, twist it, and accept only if W1 + `well_formed()` name
            # the intended label; anything else falls through to the solve.
            # `theta_twist` is a DECLARED optimization (stage 2) rather than
            # part of the retired zoo: it produces the same tower and its
            # advantage grows with charge depth, and — the point that earns it
            # its place — it extends the reach of the chart memo, since ONE
            # cached undressed tower covers its whole `e`-cone.  Still guarded
            # exactly as before (W1 + `well_formed()` naming the intended label,
            # else honest fall-through to the solve): the twist law is measured,
            # not proved.  `constructive_routes=True` also enables it, so the
            # historical combined switch keeps its old meaning.
            src = None
            if "theta_twist" in self.optimizations or self.constructive_routes:
                try:
                    src = self._twist_source(m, e)
                except NotImplementedError:
                    src = None    # m̄ undefined (central direction) — no line
            if src is not None:
                src_g, k = src
                try:
                    base = self._base_tower(src_g)
                    tw = matter_theta_twist(base, k)
                    cert = tw.well_formed()
                    if cert and tuple(cert[0]) == (tuple(m), tuple(e)):
                        self._route[g] = f"twist[k={k} from {src_g[1]}]"
                        self._base[g] = tw
                        return tw
                except Exception:
                    pass          # honest fall-through to the guarded solve
            self._route[g] = "solve"
            # WIDEN THE BOX before giving up.  `solve_level` documents that the box
            # is "a CUTOFF, never a selector", and that an `inconsistent` verdict is
            # "its documented too-narrow signature — the true answer lies outside".
            # So a `DivisibilityFailure` at the default `pad` is a statement about
            # the *window*, not about the label, and raising it reports a buildable
            # line as unbuildable.  Measured at `Spin(5)+Adj` (2026-07-28):
            # `m = (1,0)` raises at `pad=1` and **builds in 1.6 s at `pad=2`**,
            # `certify_canonical` ✓.  Enlarging is the safe direction (shrinking can
            # silently select), so escalate and only then honest-fail.
            # COST-AWARE: only retry when the failed attempt was CHEAP.  Widening
            # re-runs the whole per-μ-sector solve at a larger box, so on an
            # expensive label the retries multiply an already-long failure — at
            # `G₂+Adj`, `m=(1,0)`, the per-sector cost grows ~1.7× per sector
            # (13.2 s, 21.1 s, 37.0 s, …) and one pad=1 attempt over its 11 sectors
            # projects to ~105 min, so blind escalation would turn that into hours.
            # Where the escalation actually earns its keep the base attempt is
            # cheap: `Spin(5)+Adj` `m=(1,0)` fails at pad=1 in 1.3 s and builds at
            # pad=2 in 1.6 s.  So gate on the observed cost rather than retrying
            # unconditionally.
            import time as _time
            F = None
            last = None
            for extra in range(self._pad_escalate + 1):
                _t0 = _time.time()
                try:
                    F = solve_canonical_matter_vec(
                        self.datum, self._matter, m, e,
                        pad=self._pad + extra, lines=self.lines)
                    if extra:
                        self._route[g] = f"solve[pad+{extra}]"
                    break
                except DivisibilityFailure as ex:
                    last = ex
                    if _time.time() - _t0 > self._pad_retry_budget:
                        self._route[g] = f"solve[pad+{extra}, no retry: cost]"
                        raise DivisibilityFailure(
                            f"{self.datum.name}: L_{{{m},{e}}} unpinned at "
                            f"pad={self._pad + extra} after "
                            f"{_time.time() - _t0:.0f}s — NOT retrying at a wider "
                            f"box, because a retry re-runs the whole per-sector "
                            f"solve and would multiply an already-expensive "
                            f"failure.  Before paying for a wider `pad`, note "
                            f"that `pad` dilates the numerator box along root "
                            f"directions only and does NOT touch the 𝖖-window, so "
                            f"it cannot fix every cutoff deficiency (ruling "
                            f"D35).  `audit=` on "
                            f"`matter_star_bubbling.solve_level` localizes which "
                            f"one it is, given a candidate from the flow or from "
                            f"product-and-peel.  Last: {ex}") from ex
            if F is None:
                raise DivisibilityFailure(
                    f"{self.datum.name}: L_{{{m},{e}}} still unpinned after "
                    f"widening the box to pad={self._pad + self._pad_escalate} "
                    f"(started at {self._pad}).  DO NOT read this as 'raise the "
                    f"pad further' without evidence: `pad` dilates the numerator "
                    f"box along root directions ONLY, so it cannot reach a "
                    f"deficiency in the 𝖖-window, nor close a gap that grows "
                    f"faster than one lattice step per unit — both of which were "
                    f"real defects here (see ruling D35).  DIAGNOSE INSTEAD: get "
                    f"a candidate from an independent construction "
                    f"(`GMatterOverPure.rg_chart`, or product-and-peel) and pass "
                    f"it as `audit=` to `matter_star_bubbling.solve_level`; it "
                    f"reports whether the answer is even IN the ansatz "
                    f"(denominator / box / 𝖖-window / palindromicity) and, if it "
                    f"is, which equations it violates.  Last: {last}")
            percell: dict = {}
            for k, lvl in F.items():
                for c, v in lvl.items():
                    percell.setdefault(tuple(c), {})[tuple(k)] = v
            res: dict = {}
            for c, coeffs in percell.items():
                q, ok = divide_by_Z_vec(self.datum, self._matter, c, coeffs)
                if not ok:
                    raise DivisibilityFailure(
                        f"{self!r}: L_{{{m},{e}}} cell {c} is not divisible by "
                        f"Z_{c} — the built RG image is not of the form Z·Q")
                for k, v in q.items():
                    v = v.simplify()
                    if not v.is_zero():
                        res.setdefault(c, {})[tuple(k)] = v
            x = self.torus().element(res)
            self._base[g] = x
        return x

    # ----- chart memo + persistence (the tier's hooks) ------------------
    def chart_cache(self) -> dict:
        """The live `{(m, e): MatterWRQTorus}` memo of **neutral towers** —
        `_base`, what `_base_tower` fills.

        Keyed by the GAUGE label alone, not by the full `((m, e), w)`: the
        flavour irrep multiplies the neutral tower by `χ_w(μ)` written out on the
        substrate (`chart` above), so caching per `w` would store the same tower
        many times over.  One tower per gauge label is the whole content."""
        return self._base

    def route_cache(self) -> dict:
        return self._route

    def nested_cache_algebras(self) -> dict:
        """The inner pure-`G` algebra's charts travel in the same file.

        `decompose` reads its lowest μ-level slice with `self._pure` (ruling D3,
        `Q[0⃗] = pure`), so a matter cache that did not carry the pure charts
        would still pay to rebuild them on the first read — the exact cost the
        file exists to avoid."""
        return {"pure": self._pure}

    def _verify_loaded_chart(self, label: Label, x) -> dict:
        """W1 + the W2 read naming THIS gauge label, on a tower loaded rather
        than built.

        Weaker than the pure tier's battery, and the reason is stated rather than
        papered over: the five-condition battery `star_bubbling.verify_axioms`
        runs is pure-gauge — its support bound is `conv(W·m)` on the gauge
        lattice and its (★) tests `criterion` against `R(G)`, neither of which is
        the matter tier's condition (the matter element is the quotient `Q` of
        `F = Z·Q`, graded by μ-level).  So what is checked here is what is
        *exactly* checkable on `MatterWRQTorus`: whole-element bar-invariance
        across all μ-levels, and the base-level slice reading back precisely the
        label the entry is filed under — which is what catches a chart stored
        against the wrong label, the failure a cache can actually introduce.
        The matter analogue of the full battery is a real piece of work
        (`matter_star_bubbling` has no `verify_axioms`) and is recorded as open
        rather than approximated here."""
        wf = x.well_formed()
        out = {"bar (W1)": (bool(x.well_formed_w1()), "")}
        ok = wf is not False and tuple(wf[0][0]) == tuple(label[0]) \
            and tuple(wf[0][1]) == tuple(label[1])
        out["seed (W2) names the label"] = (
            ok, f"well_formed() = {wf}, filed under {label}")
        return out

    def chart(self, label: Label):
        """`L_label` as a `MatterWRQTorus`: `L_{(g,w)} = χ_w(μ)·L_{(g,1)}`.

        `χ_w` is expanded into its Cartan weights (`_flav_levels`) and each
        weight shifts a copy of the neutral tower by that μ-level — the
        character multiple written out on the substrate.  At the trivial irrep
        this is just the neutral tower."""
        g, w = self._split(label)
        x = self._base_tower(g)
        levels = self._flav_levels(w)
        if levels == {(0,) * self._M: 1}:
            return x
        acc = None
        for k, mult in levels.items():
            piece = x._scaled(LaurentPoly({0: int(mult)}), k)
            acc = piece if acc is None else acc + piece
        return acc

    def _attribute(self, x, max_rounds: int = 512) -> dict:
        """`{(gauge label, μ-level k⃗): LaurentPoly}` — level-ascending read on
        the substrate, before any flavour packaging.

        At the globally lowest μ-level the slice is a combination of **pure**
        charts — that is D3 (`Q[μ⁰] = pure`), so the read has no target and
        cannot fabricate — decoded by the pure-`G` WRQ engine; the matching
        level-shifted neutral towers are subtracted and the remainder recurses.
        Honest-fails when it does not terminate or when the lowest slice is not
        in the pure span."""
        cur = x
        out: dict = {}
        for _ in range(max_rounds):
            fam = cur.to_family()
            if not fam:
                return {k: C for k, C in out.items() if not C.is_zero()}
            k0 = min(fam, key=lambda k: (sum(k), k))
            got = _wrq_decompose(
                self.datum, fam[k0],
                build=lambda dat, mv, ev: self._pure.chart(
                    self._pure.fold(mv, ev)))
            if not got:
                raise NotImplementedError(
                    f"{self!r}.decompose: μ-level {k0} slice is non-zero but "
                    f"decodes to nothing in the pure basis — off scope")
            for (mv, ev), C in got.items():
                g = self._pure.fold(mv, ev)
                key = (g, tuple(k0))
                out[key] = out.get(key, LaurentPoly.zero()) + C
                cur = cur + self._base_tower(g)._scaled(
                    C * LaurentPoly({0: -1}), k0)
        raise NotImplementedError(
            f"{self!r}.decompose: attribution did not terminate in "
            f"{max_rounds} rounds — off scope")

    def decompose(self, x, max_rounds: int = 512) -> Element:
        """Canonical read: substrate attribution, then **flavour packaging** —
        per gauge label and per 𝖖-order, the μ-graded content is un-branched
        into `∏_i U(n_i)` characters (`_flav_unbranch`).

        The 𝖖-order-by-𝖖-order split matters: a structure constant is a
        `Z[𝖖,𝖖⁻¹]`-combination of flavour irreps, so the flavour content must be
        a genuine character combination **at each order in 𝖖 separately**.  If
        it is not, `sun_characters.decompose` honest-fails rather than
        fabricating a weight."""
        attributed = self._attribute(x, max_rounds=max_rounds)
        # {gauge label: {qexp: {k⃗: int}}}
        by_gauge: dict = {}
        for (g, k), C in attributed.items():
            for e, c in C._coeffs.items():
                if c:
                    row = by_gauge.setdefault(g, {}).setdefault(e, {})
                    row[k] = row.get(k, 0) + int(c)
        out: dict = {}
        for g, byq in by_gauge.items():
            for e, levels in byq.items():
                for w, mult in self._flav_unbranch(levels).items():
                    if not mult:
                        continue
                    lab = (g, w)
                    out[lab] = (out.get(lab, LaurentPoly.zero())
                                + LaurentPoly({e: int(mult)}))
        return Element({lab: C for lab, C in out.items() if not C.is_zero()})

    # ----- trace / pairing: package the μ-levels into characters ----------
    def _package(self, levels_by_q: dict, K: int) -> RPowerSeries:
        """`{qexp: {k⃗: int}} → RPowerSeries` over `∏_i U(n_i)`.

        The tier's default `_flavour_element` maps ONE μ-level to ONE ring
        basis key, which is right for an abelian flavour ring and wrong here —
        un-branching needs every level at a given 𝖖-order at once.  So `trace`
        and `inner_product` are overridden to collect first and package after."""
        acc: dict = {}
        for e, levels in levels_by_q.items():
            if not (0 <= e <= K):
                continue
            terms = self._flav_unbranch(levels)
            if not terms:
                continue
            el = RElement(self._R, {w: c for w, c in terms.items() if c})
            acc[e] = el if e not in acc else (acc[e] + el)
        return RPowerSeries(self._R, acc, K)

    def _levels_by_q(self, tr, K: int) -> dict:
        out: dict = {}
        for lev, lp in self._trace_levels(tr).items():
            k = tuple(int(x) for x in lev)
            for e, c in lp._coeffs.items():
                if 0 <= e <= K and c:
                    row = out.setdefault(e, {})
                    row[k] = row.get(k, 0) + int(c)
        return out

    def trace(self, a: Label, K: int = 20) -> RPowerSeries:
        """`Tr(L_a)`, packaged over `∏_i U(n_i)`.

        Goes through `AbeKAlgebra._chart_trace`, which ties the matter **Nahm window
        `W`** to `K`.  Calling `chart.trace(K=K)` directly — as this did before
        2026-07-28 — leaves `W` at its default 4, so the result is silently wrong
        above `𝖖⁴` (the first bad order is exactly `W+1`, measured).  This override
        exists for the flavour *packaging*, so it must not re-introduce the
        truncation the base class fixes."""
        return self._package(
            self._levels_by_q(self._chart_trace(self.chart(a), K), K), K)

    def inner_product(self, a: Label, b: Label, K: int = 20) -> RPowerSeries:
        """`I_{a,b} = Tr(ρ(L_a)·L_b)` — chart-side, packaged over the flavour
        ring.  Same Nahm-window routing as `trace` above."""
        prod = self.chart(a).rho() * self.chart(b)
        return self._package(self._levels_by_q(self._chart_trace(prod, K), K), K)

if __name__ == "__main__":
    import root_datum as rd

    # Kept deliberately cheap: the pairing is the expensive read (the Schur
    # residue window), so only the rank-1 row prints it.  The full sweep — all
    # four theories, both certification legs, the whole derived API — is
    # the suite in the source repository.
    for tag, datum, matter, kw, lab, pair in [
        ("SU(2)+2×2", rd.su_2(), (1,), {"nf": 2}, ((1,), (0,)), True),
        ("Sp(4)+2×4", rd.sp_n(2), (1, 0), {"nf": 2}, ((1, 0), (0, 0)), False),
        ("Spin(5)+2×4ˢ", rd.b_n_simply_connected(2), (0, 1), {"nf": 2},
         ((1, 1), (0, 0)), False),
    ]:
        A = GNAbeKAlgebra(datum, matter, **kw)
        a = (lab, (0,) * A.M)
        print(f"{tag}: {A!r}")
        print(f"   shape      = {A.torus_shape()}")
        print(f"   L_{a}: W1 = {A.verify_chart_bar(a)}, "
              f"certify = {A.certify_canonical(a)}")
        print(f"   ρ(L_a)     = {A.rho(a)}")
        if pair:
            print(f"   Tr(1, K=4) = {A.trace(A.identity(), 4)}")
            print(f"   I(a, a)    = {A.inner_product(a, a, 4)}")
