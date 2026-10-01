"""Search the ORDER, not the element: a BFS over BPS-factor insertion points.

`bps_factor_spectrum.py` builds `S` in *one* order, whichever the caller asks for.  `S`
does not depend on that choice, but the **factorisation** does — the factor content
`Ω(γ, s)` is a function of the order, and some orders factor `S` far more simply
than others.  This module searches for a simple one.

THE GOAL

> *"I think we will not find easy greedy heuristics when building `S` step by
> step.  A \\*good\\* move would be recognized by the fact that subsequent moves
> are easier.  But we could have BFS search method which explores different
> factor ordering and selects for simpler factorizations (smaller `s`, smaller
> `Ω`, etc.) in the hope of landing a spec (finite product of `E_𝖖` factors)."*

"Simple" has a sharp target: **spin 0 only**, which is what consumers need.  So that is what `search()` aims at.

WHY SUPPRESSING SPIN IS THE HEURISTIC, AND WHAT A SPEC LOOKS LIKE FROM HERE

> **The author, 2026-08-13.** *"a 'spec' `S` will just stop populating at some point.
> It is generally believed that any `s>0` factor will always imply the existence
> of infinitely many more factors."* … *"hence the 'suppress s' heuristic."*

That is the justification for the whole ranking, and it is a chain worth writing
out because it makes the spin preference principled rather than aesthetic:

    an `s > 0` factor  ⟹ (believed)  infinitely many more factors
                       ⟹  the content never stops populating
                       ⟹  not a spec.

So suppressing spin is not one desideratum among several — it is the observable
proxy for termination, which is the property actually wanted.  Hence
`require_spin_zero` as an exact prune, and the `SPIN_PENALTY` in the cost for
when that prune is off.  The implication is a **belief**, not a theorem, and
nothing here proves it; it is why the heuristic is expected to work, not evidence
that it did.

Correspondingly the acceptance signal is **stopping**, not spin alone:
`Result.stopped` reports that factors ceased appearing *strictly below* the
cutoff, so the recursion was solved through degrees the factorisation did not
reach.  Measured, the separation is stark and it grows with the cutoff — a good
order's support stays constant in `D` while a bad one's tracks the cone
(Kronecker-3, `D = 6 → 12`: 2 charges throughout, against 17 → 68 of 27 → 90 cone
points, i.e. 63 % → 76 % of the cone).  That is the author's *"a bad `S` will
typically populate `Ω` on the whole positive cone, a good `S` will populate very
sparse `Ω`"*, and it is why the search separates them easily.

EVERYTHING HERE IS UP TO A CONE CUTOFF, AND `S` IS NEVER "DONE"

> **The author, 2026-08-13.** *"Note that the S-builder will never be 'done' typically.
> Instead, if you want to know `S` up to some cutoff in the positive cone, you
> solve the recursion up to that cutoff."*

So a result reporting `is_spec` says: **`S` is the ordered product of these `E_𝖖`
factors on the cone up to degree `D`, they stopped appearing strictly before `D`,
and the product still reproduces `S` on a STRICTLY WIDER cone.**  That is as
close to "the factorisation terminates" as solving a recursion to finite `D` can
get, and it is deliberately not the same claim: a larger `D` could always reveal
more.  Read `Result.spec` as "the factors seen within the cutoff", and raise the
cutoff to see further.  (The same caveat applies to
`recursive_spectrum.extract_spec_from_quiver`, which verifies its spec in-cone
only.)

The third clause is there because the first two are **measurably not enough**.
An in-cone rebuild plus a stopping margin admits *impostors*: contents that agree
with `S` at every degree they were checked at and disagree at the next one.  Two
are pinned in the tests (`STOP_MARGIN`, `CONFIRM_EXTRA` and
`test_a_narrow_search_can_return_an_in_cone_impostor` carry the numbers); the
sharp one is the 5-cycle at `D = 6`, where a 9-factor impostor and the valid
8-factor answer top out at the **same** degree, so no margin whatsoever can
separate them and only the wider-cone rebuild does.

THE COST IS AN OVERALL ONE

> **The author, 2026-08-13.** *"with looking ahead I did not just mean to the next move.
> more like an overall 'cost' given by the number of terms with a dislike for
> `s>0` as well."*

So `default_cost` counts terms — `terms + SPIN_PENALTY · Σ|a_s|·2s` — over
everything placed **plus everything already forced at the next cone degree**.
Those forced terms are not a lookahead bolted on: `_readout` fixes them from what
has been placed, nothing later can change them, so they are as much a property of
the candidate as its own factors and belong inside the count.  Keeping them there
is also what makes the cost able to separate *siblings* — children of one parent
share their newest degree's content, so a count of placed terms alone cannot.
See `default_cost`, which records the two earlier versions and how each was
measured wrong.

WHAT IS PRUNED EXACTLY, AND WHAT IS ONLY RANKED

* **Exact, and the reason the search is selective at all**: a branch dies the
  moment a forced `Ω` violates the requirement — a nonzero spin under
  `require_spin_zero`, a negative multiplicity under `require_positive`.  This is
  a *decision*, not a heuristic: `Ω` is forced, so a violating branch can never
  become a spin-0 factorisation later.  Note these are the *hard* form of the
  same preference the cost expresses softly; turn them off to let the cost do the
  work across the general factorisation space.
* **Heuristic, and only used to rank**: `keep_per_degree`, the number of candidate
  orders carried forward from each cone degree (the search literature calls this a
  beam width).  Discarding the rest can drop the branch that would have won, so a
  run that finds nothing is *not* a proof that nothing exists —
  `Result.exhaustive` records whether that cap or the branch cap actually bit, and
  only when neither did is the negative answer complete over the searched space.

RELATION TO THE OTHER TWO ROUTES, which this does not replace:

* `recursive_spectrum.extract_spec_from_quiver` consumes an already-built `S` and
  recovers a spec from it by insertion.  Cheaper, and it is the right tool when
  any finite chamber will do.  This module is for asking *which* orders give one,
  and for the quivers where the default order gives a bad factorisation.
* `bps_quiver_tools.BPSQuiver.find_negating_sequence` is the **cluster-side**
  spec finder — bidirectional green-sequence BFS, the cluster definition of the
  DT invariant, explicitly retained.  It searches mutation
  sequences; this searches factor orders.  They are different searches for
  related objects, and agreement between them is evidence, so it is checked
  rather than assumed.

FACTORING AN `S` YOU ALREADY HAVE

> **The author, 2026-08-13.** *"We should also look for an algorithm to find the
> simplest factorization of an already-computed `S`.  Simplifying could be useful
> even at intermediate stages, before pushing a cutoff higher."*

`simplify_factorisation(S, pairing, nodes, cutoff)` is the same search entered
from the other end.  The engine asks an order for exactly one number per charge
— the prescribed `𝖖¹`-coefficient — so **an `S` that exists already carries its
own leading data** (`leading_data_from_spectrum`, derived and controlled against
`BPSFactorSpectrum.target` at 54 charges over three quivers, spin included).  Nothing
about where the `S` came from needs to be known.

`verify=True` by default, and it earns its cost: handed 3-cycle(2,1,1)'s `S`
against 3-cycle(1,1,1)'s quiver, `verify=False` returns a **valid 4-factor spec —
of 3-cycle(1,1,1)**, silently, because the two share their leading data.  A tool
whose job is to take `S` on trust has to check that trust.

For the *"intermediate stages"* half, `seed_order=` accepts a previous
`Result.piece_key()` and replays it as far as it stays clean, then hands over to
the BFS.  Measured 1.15–2.2× against a cold search on the quivers where the
search is the expensive part, always landing on the identical factorisation.  The
seed is **donated, not trusted** — necessarily, since replaying a good low-cutoff
order verbatim at a higher cutoff was measured to fail 4 times in 5.

Against the repo's previous route for this (`recursive_spectrum.Theory.
extract_spec_insert`, an insertion DFS): identical answers, and the gap is in
cost — 0.38 s against 7.87 s at the 4-cycle, and the DFS exceeds a 180 s box at
the 5- and 6-cycles where this returns in 2.9 s and 21.3 s.  That comparison also
turned up a live defect in the DFS's default cutoff.

NAME.  `factor_order_search` / `FactorOrderSearch` — ratified by the author
2026-08-13 ("Good naming").  The unit it orders is the **BPS factor**
`E^{(s)}_𝖖(X_γ)^{Ω(γ,s)}` (the author's name), one per pair `(γ, s)`.

Nothing here says "ray", under the author's later and stronger rule: *"rays are only
a viable notation in the presence of a `Z_γ`"* — a ray is a ray in the
central-charge plane, and the default order is phase-free, so there is no `Z_γ`
and there are no rays.  The per-charge machinery this calls is named for the
charge accordingly (`order_key`, `charge_series`, `_charge_multiply`).  That rule
supersedes the narrower 2026-08-13 one, which had allowed "ray" for anything
per-charge.

SURFACE.  `find_simple_factorisation` (quiver in) and `simplify_factorisation`
(`S` in) stay **two separate module-level functions**, and the search gets no
`BPSKAlgebra` entry point ("leave them as separate
methods"), asked because adding one would be a public-surface addition on a spine
class.  They are the same search entered from opposite ends and were not merged.

Run:  PYTHONPATH=. python factor_order_search.py
"""

from __future__ import annotations

import bisect
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from habiro import HabiroElement
from bps_factor_spectrum import (
    BPSFactorSpectrum, Vec, acyclic_node_order, expansion, nahm_generators,
    spin_decompose,
)

H0 = HabiroElement.zero()
H1 = HabiroElement.one()


# --------------------------------------------------------------------------
# an already-computed `S` is its own leading data
# --------------------------------------------------------------------------


def leading_data_from_spectrum(
        spectrum: dict[Vec, HabiroElement]) -> Callable[[Vec], int]:
    """Read a computed `S`'s leading data back off it, for re-factorisation.

    The engine asks an order for exactly one number per charge — `target(k)`,
    *"the prescribed `𝖖¹`-coefficient of `S` at this charge"* — and everything
    else about the factorisation is forced from it.  So an `S` that already
    exists carries its own leading data, and factoring it needs no separate
    prescription and no assumption about where it came from.

    DERIVED, NOT ASSUMED, and controlled before use: over the pentagon, pure
    SU(2) at weak coupling (where `Ω` carries spin, `𝖖⁻¹ + 𝖖` at the W boson)
    and 3-cycle(1,1,1), `expansion(S[γ])[1] == BPSFactorSpectrum.target(k)` at **all
    54 cone charges, zero mismatches**.  The spin case is the one that matters:
    it rules out reading the whole element where only the `𝖖¹` coefficient is
    prescribed.

    A charge `S` does not mention has leading datum 0 — `spectrum_generator()`
    stores only nonzero entries, so absence is a zero and not a gap.
    """
    def leading(charge: Vec) -> int:
        element = spectrum.get(tuple(charge))
        return 0 if element is None else expansion(element).get(1, 0)

    return leading


# --------------------------------------------------------------------------
# a placed factor, and the trace-monoid normal form that dedupes orders
# --------------------------------------------------------------------------

#: One placed BPS factor `E^{(s)}_𝖖(X_γ)^{a_s}`: `(node coords, 2s, a_s)`.
#: The cone degree is `sum(k)` and is not carried.
Piece = tuple[Vec, int, int]


def _normal_form(placed: Sequence[Piece],
                 commutes: Callable[[Vec, Vec], bool]) -> tuple[Piece, ...]:
    """Lexicographic normal form of a placement, modulo commuting factors.

    Two BPS factors commute **exactly** when their charges pair to zero:
    `A_γ(Ω)` is a series in `X_{nγ}` and `A_δ(Ω')` one in `X_{mδ}`, and
    `X_{nγ}X_{mδ} = 𝖖^{nm⟨γ,δ⟩}X_{nγ+mδ}`, so `⟨γ,δ⟩ = 0` makes every pair of
    terms commute and nothing else does.  Placements therefore form a
    **trace monoid** (Mazurkiewicz), and the product depends only on the
    trace-equivalence class — so two orders in the same class are the same state
    and only one of them is worth carrying.

    The normal form is the usual one: repeatedly swap an adjacent out-of-order
    commuting pair until no swap applies.  Integer work only, no arithmetic on
    the accumulated element.

    ⚠ **Sound in the safe direction, and deliberately not complete.**  Distinct
    trace classes can still give the same product by accident; this merges none
    of those, so it never conflates two genuinely different states — it only
    misses a merge, which costs search width rather than correctness.  Merging by
    the product itself is not available: `HabiroElement` is unhashable, and
    comparing every pair of states each layer would cost more than the duplicates
    do.
    """
    out = list(placed)
    changed = True
    while changed:
        changed = False
        for i in range(len(out) - 1):
            left, right = out[i], out[i + 1]
            if left > right and commutes(left[0], right[0]):
                out[i], out[i + 1] = right, left
                changed = True
    return tuple(out)


# --------------------------------------------------------------------------
# what "simple" means
# --------------------------------------------------------------------------

#: How much one unit of spin costs, relative to one term.  The default makes a
#: single `s = 1/2` multiplet cost as much as five spin-0 ones, so the search
#: trades several extra terms away to avoid a spin — the "dislike for `s > 0`" —
#: without treating spin as infinitely bad, which is what a hard prune does.
SPIN_PENALTY = 4

#: Degrees of empty cone required above the last factor before a factorisation is
#: even CONSIDERED to have stopped.  A screen, not a certificate — see
#: `CONFIRM_EXTRA`.
#:
#: **A margin of 1 is measurably useless and a margin of 2 is measurably not
#: enough.**  Margin 1: at 4-cycle(1) with `D = 4` a spin-0 content topping out
#: at degree 3 rebuilds `S` up to 4 and is wrong from degree 5 on.  Margin 2 was
#: then derived from 84 checks over seven quivers and **refuted at the eighth**:
#: cycle6, certified at `D = 7` with margin 2 and top degree 5, is valid at `D=7`
#: and **invalid at `D=8`**.  cycle5 at margin 2 *is* valid two degrees up, so it
#: is not a rank-monotone rule either.
#:
#: The lesson is not "use 3".  A constant fitted to a sample is what failed
#: twice; the margin is a *proxy* for termination and the honest certificate is
#: to verify at a larger cone, which is what `CONFIRM_EXTRA` does.  This is kept
#: only as the cheap screen that decides whether the expensive confirmation is
#: worth running.
STOP_MARGIN = 2

#: Extra cone degrees the found factorisation is REBUILT at before `is_spec` is
#: granted — i.e. the **declared verification depth**, not a certificate.
#:
#: ⚠ **THERE IS NO SUFFICIENT CONSTANT HERE, AND THREE HAVE NOW BEEN REFUTED.**
#: `STOP_MARGIN = 1`, `STOP_MARGIN = 2` and `CONFIRM_EXTRA = 1` each looked
#: adequate on a sample and each was defeated by a factorisation that agrees with
#: `S` everywhere the check looked and disagrees one degree past it:
#:
#:   * margin 1 — 4-cycle at `D = 4`, wrong from degree 5;
#:   * margin 2 — cycle6 at `D = 7`, wrong at 8; and the 5-cycle at `D = 6`,
#:     where a valid answer and a wrong one share a top degree, so *no* margin
#:     separates them;
#:   * this check at depth 1 — cycle6 at `D = 7`, **seed 1**: certified (so
#:     verified at 8), exact at 7 and 8, and **wrong at 9 and 10**.
#:
#: So `2` is not the fix.  The lesson, stated once and not re-learned: a check
#: that stops at a fixed depth is defeated by a counterexample that lives past
#: that depth, and no amount of sampling finds the one you did not reach.
#: Accordingly this constant no longer pretends to settle anything — it is the
#: depth a run *paid for*, `Result.confirmed_to` reports how far that reached,
#: and `is_spec` is a claim relative to that degree rather than a global one
#:  A caller who needs more buys it with
#: `confirm_extra=`.
#:
#: Worth recording how the third refutation was missed at first: a hunt over 11
#: quivers × 6 cutoffs × 2 widths found 95 certified factorisations all still
#: exact two degrees up, and reported no counterexample.  It varied the width and
#: held the **seed** fixed — and the seed is the axis that decides what a run
#: draws (A55).  Silence from a scan is evidence only about the axes it varied.
CONFIRM_EXTRA = 1

#: Restarts (fresh position samples at the same width) tried before the search
#: reports no factorisation.  See `FactorOrderSearch._escalate` for the
#: measurement: a run is a *draw*, and at one fixed width the draws span
#: no-spec / valid-spec / refused-candidate.
#:
#: Unlike `STOP_MARGIN` this is not a calibrated threshold and cannot be wrong in
#: the same way — restarts are **monotone** (more of them can only find more,
#: never less), so this is a cost dial.  Raising it buys coverage linearly in
#: time; lowering it only loses answers, never admits wrong ones, since every
#: attempt still passes `CONFIRM_EXTRA`.
RESTARTS = 4


@dataclass(frozen=True)
class Content:
    """The summary of a factorisation that a cost function ranks.

    Every field is a property of the factor content `Ω`, i.e. of the ORDER — none
    of them is a property of `S`, which is order-independent.
    """

    factors: int               #: charges carrying a nonzero `Ω`
    terms: int              #: `Σ_{(γ,s)} |a_s|` — the number of terms
    spin_weight: int        #: `Σ_{(γ,s)} |a_s|·2s` — how much of that is spin
    max_spin_doubled: int   #: `max 2s` over the placed multiplets
    negatives: int          #: how many `a_s < 0` (legal, but not spec-like)
    forced_next: int        #: weighted terms already FORCED at the next degree

    @property
    def is_spin_zero(self) -> bool:
        return self.max_spin_doubled == 0


def weighted_terms(terms: int, spin_weight: int,
                   spin_penalty: int = SPIN_PENALTY) -> int:
    """`terms + spin_penalty · Σ|a_s|·2s` — the number of terms, disliking spin."""
    return terms + spin_penalty * spin_weight


def default_cost(content: Content):
    """The author's cost: **the number of terms, with a dislike for `s > 0`.**

        cost  =  weighted_terms(placed)  +  weighted_terms(forced at the next degree).

    **One overall count, not a lookahead bolted on.**  The `Ω` at the next cone
    degree is already *forced* by what has been placed — nothing later can change
    it — so those terms are as much a property of this candidate as the ones it
    has placed, and they belong inside the count rather than beside it as a
    tiebreak.  That is what makes the cost overall while keeping the predictive
    content: a candidate that has few terms so far but has already committed
    itself to many is correctly ranked worse than one that has not.

    Two earlier versions are recorded because each was measured wrong:

    * ranking primarily by the **next degree alone** (a one-step lookahead) is
      the version the author corrected — it is blind to how the candidate got here,
      and states from different parents differ exactly there.
    * ranking by **placed terms alone**, with the next degree demoted to a
      lexicographic tiebreak, loses the SU(3)-cyclic 6-factor spec at the default
      width: fewer terms so far does not predict fewer later, and the tiebreak
      fires too rarely to correct it.

    The remaining keys, after the cost:

    * `negatives` — negative multiplicities are legal (the integral palindromic
      lattice survives what positivity does not) but a spin-0 `E_𝖖` product
      has none.
    * `factors` — fewer charges carrying content, as a last resort.
    """
    return (weighted_terms(content.terms, content.spin_weight)
            + content.forced_next,
            content.negatives, content.factors)


@dataclass
class _State:
    placed: list[Piece]
    prefix: list[dict]              #: `prefix[i]` = product of `placed[:i]`
    prefix_deg: list[dict]
    omega: dict[Vec, dict[int, int]]
    content: Content | None = None
    pending: int | None = None
    """Index from which the product is not yet built, or `None` if it is.

    Set only on the last cone degree, where the ranking provably needs no
    product; `_materialise` finishes the arithmetic for the state that wins."""

    lookahead: tuple | None = None
    """`(degree, readout)` memo — the scoring lookahead IS the next layer's
    readout, so computing it twice is pure duplication of the dominant cost."""

    @property
    def product(self) -> dict:
        return self.prefix[-1]


@dataclass
class Result:
    """What a search found, and how much of the space it actually covered."""

    order: list[Vec] = field(default_factory=list)
    """The charges in placement order, one entry per placed multiplet."""

    omega: dict[Vec, dict[int, int]] = field(default_factory=dict)
    spectrum_generator: dict[Vec, HabiroElement] = field(default_factory=dict)
    content: Content | None = None
    is_spec: bool = False
    """Spin-0, stopped, and verified equal to `S` up to `confirmed_to`.

    ⚠ **Read the degree, not the boolean.**  This is not a certificate that the
    factorisation is correct everywhere: three fixed-depth checks have been
    refuted (see `CONFIRM_EXTRA`), most recently one that passed at depth 1 and
    was wrong two degrees up.  `True` means "checked that far and it held".

    The original wording follows, still accurate about the in-cone part:

    Not a claim that the factorisation terminates: the recursion is solved to a
    finite cone degree `D`, so what is established is that `S` restricted to that
    cone is the ordered product of these `E_𝖖` factors and that no further factor
    appeared before `D`.  Raise the cutoff to see
    further.
    """

    spec: list[Vec] | None = None
    """Those factors as a charge list (`a_0` copies of each), or `None`.

    The factors seen WITHIN the cutoff, in placement order — see `is_spec`.
    """

    exhaustive: bool = True
    """Neither cap bit, so a negative answer is complete over the searched space."""

    top_degree: int = 0
    """The largest cone degree at which any factor was placed."""

    margin: int = 0
    """`cutoff − top_degree`: how many degrees of empty cone sit above the last
    factor.  The evidence behind `stopped`, exposed so a caller can judge it."""

    stopped: bool = False
    """Factors stopped appearing, with at least `STOP_MARGIN` degrees to spare.

    The in-cone form of "this factorisation terminates": the recursion was
    solved through degrees the factorisation did not reach, so the stopping is
    observed rather than cut off by the truncation.

    ⚠ **The margin has to be at least 2, and that is a measurement, not a
    convention** — see `STOP_MARGIN`.  A margin of 1 was tried first and is
    unsound: at 4-cycle(1) with `D = 4` it accepts a factorisation that rebuilds
    `S` in-cone and is wrong from degree 5 on.
    """

    confirmed_to: int = 0
    """Largest cone degree at which the product was CHECKED equal to `S`, or 0.

    The honest content of `is_spec`, and the reason it is a number rather than a
    boolean: three fixed-depth checks have been refuted here (`STOP_MARGIN` at 1
    and at 2, `CONFIRM_EXTRA` at 1), each by a factorisation exact everywhere the
    check looked and wrong one degree past it.  No finite depth certifies, so what
    a result can honestly report is how far it was verified — `S` equals this
    ordered product on the cone up to `confirmed_to`, and nothing is claimed
    above.  Raise `confirm_extra` to move it.
    """

    optimal: bool = False
    """The factorisation attains the provable lower bound on the factor count."""

    route: str = "search"
    """Which route answered: `"strip"` (the acyclic short-circuit), `"search"`
    (the BFS under the spin-0 prune), `"search-wide"` (the BFS after the width
    escalation — the first pass found nothing and a wider beam did), or
    `"search-sparse"` (the fallback pass with the prune dropped, which returns
    the sparsest order found on a quiver that admits no spin-0 factorisation)."""

    content_verified: bool | None = None
    """The whole content rebuilds `S` in-cone by the independent sum route.

    Reported on the `"search-sparse"` route, where `spec` is `None` and
    `is_spec` is `False` because the content carries spin — so neither of those
    says anything about whether the answer is *right*.  `None` elsewhere: the
    spec routes already carry the stronger `is_spec`, which subsumes this.

    It became checkable only when the Nahm shift gained its diagonal `n²`
    (`nahm_local.general_nahm_habiro`).  Before that this route was the module's
    one unverified output.  Measured `True` at Markov(2,2,2) and at
    3-cycle(3,3,3), the latter carrying spin up to `2s = 5`.
    """

    states_examined: int = 0
    branches_cut: int = 0

    def piece_key(self) -> Callable[[Vec, int], object]:
        """The found order, as a `piece_key` for `bps_factor_spectrum.BPSFactorSpectrum`.

        The point of the search is to be *reusable*: the winning order becomes an
        ordinary argument to the engine, so a caller can rebuild `S` in it, feed
        it to `BPSKAlgebra.spectrum_generator_from_factors`, or cross-check it —
        rather than the search owning a private build path.  Pairs not placed by
        the search sort last, in their own natural order, so the key is total on
        the whole cone and not only on what was seen.
        """
        rank = {}
        for position, (k, two_s, _amount) in enumerate(self._pieces):
            rank.setdefault((tuple(k), two_s), position)

        def key(k: Vec, two_s: int):
            # UNIFORM TUPLE SHAPE, and it has to be: a placed pair returns its
            # position and an unplaced one has to sort after every placed pair,
            # but returning a bare `int` for the first and a tuple for the second
            # makes the engine's `bisect` compare `int` against `tuple` and
            # raise.  Nothing hits it while the order is replayed at the cutoff
            # that produced it — every pair is known — so it only appears when
            # the order is carried to a LARGER cutoff, which is exactly the use
            # the method exists for.
            placed = rank.get((tuple(k), two_s))
            if placed is not None:
                return (0, placed, ())
            return (1, 0, (tuple(k), two_s))

        return key

    _pieces: list[Piece] = field(default_factory=list, repr=False)


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------

class FactorOrderSearch:
    """BFS over BPS-factor insertion points, ranked for simple factorisations.

    Parameters
    ----------
    pairing, node_charges, cutoff
        As for `bps_factor_spectrum.BPSFactorSpectrum`; `cutoff` is the cone-degree
        truncation and the only truncation (never a `𝖖`-truncation).
    leading_data
        Optional `charge -> int`; default is the BPS one, `−1` on the nodes.
    keep_per_degree
        How many candidate orders are carried forward from each cone degree; the
        rest are discarded.  (The search literature calls this a beam width.)
        `None` keeps all — exhaustive over the branch caps below, and the only
        setting whose negative answer is a proof over the searched space.

        **The default 4 is measured, load-bearing in both directions, and
        re-measured after the cost model and the two-phase scoring changed** —
        the numbers below are from the current engine, over seven quivers:

            keep   result                                     cycle7 D=8
              2    LOSES SU(3)-cyclic and cycle7; cycle5 8->9     8.8 s
              4    every answer correct                         11.8 s
              8    every answer correct                         30.5 s
             16    every answer correct                         55.6 s

        So 4 is the smallest width that finds everything, and widening buys
        nothing at 2.6-4.7x the cost.  Below it the search degrades two different
        ways, and both are pinned as tests: it loses a factorisation outright,
        and — the harder case to notice — it can still succeed while returning a
        *worse* one.  A run reporting `exhaustive=False` really can be hiding an
        answer, or a better one.

    branch_cap
        Placements enumerated per state per degree.

        **Measured NOT to be a cost lever, and harmful to lower.**  Sweeping
        8/16/32/64/128 over six quivers: the times are flat (cycle6 2.79 s at 8
        against 2.58 s at 128) because the cap usually does not bind — the actual
        number of position tuples for a degree's handful of pieces is already
        below it — while lowering it loses answers where it *does* bind: at 8 the
        cycle6 spec is lost entirely and cycle5 degrades from 8 factors to 9; at
        16 the SU(3)-cyclic spec is lost.  So 64 stays, and shrinking it to buy
        speed buys none.

        Positions are taken in a
        deterministic order and then sampled once that runs out, so a capped run
        is reproducible under `seed`.
    require_spin_zero
        Cut a branch as soon as a forced `Ω` carries a nonzero spin.  This is the
        prune that makes the search selective, and it is EXACT — `Ω` is forced,
        so a branch that has emitted a spin can never become spin-0 later.  Turn
        it off to explore the general factorisation space.
    require_positive
        Cut a branch as soon as some `Ω(γ, s) < 0`.  Exact in the same sense.
    fallback
        When `require_spin_zero` finds nothing, re-run with it dropped and return
        the sparsest order the cost can find instead of nothing.  On a quiver with
        no spin-0 factorisation — Markov, the wild 3-cycles — that is the
        difference between an empty answer and a usable one, and a sparse `Ω` is
        worth having in its own right because it makes downstream calculation
        cheaper.  `Result.route` reports `"search-sparse"` when
        this fired, and `is_spec` is `False`, so the weaker claim is never
        mistaken for the stronger one.  **The weaker claim is still verified**
        (2026-08-14): this content carries spin, so the spec-only rebuild could
        not reach it and it was the module's one unchecked answer until the Nahm
        shift gained its diagonal `n²`.  `Result.content_verified` now reports
        the independent rebuild.
    cost
        `Content -> sortable`, smaller is better.  Default `default_cost`.
    """

    def __init__(
        self,
        pairing: Sequence[Sequence[int]],
        node_charges: Sequence[Sequence[int]],
        cutoff: int,
        *,
        leading_data: Callable[[Vec], int] | None = None,
        keep_per_degree: int | None = 4,
        branch_cap: int = 64,
        require_spin_zero: bool = True,
        require_positive: bool = True,
        fallback: bool = True,
        cost: Callable[[Content], object] | None = None,
        seed: int = 20260813,
        seed_order: Callable[[Vec, int], object] | None = None,
        escalate: bool = True,
        confirm_extra: int | None = None,
    ):
        # One engine instance supplies the arithmetic — the cone, the readout,
        # `_forced`, the charge multiply.  The search owns the ORDER and nothing
        # else, which is what keeps it honest: any factorisation it reports is
        # one `bps_factor_spectrum` itself would produce given the same placement.
        self._engine = BPSFactorSpectrum(pairing, node_charges, cutoff,
                                   leading_data=leading_data, order="lex")
        self.rank = self._engine.rank
        self.degree_cap = self._engine.degree_cap
        self.keep_per_degree = keep_per_degree
        self.branch_cap = int(branch_cap)
        self.require_spin_zero = require_spin_zero
        self.require_positive = require_positive
        self.fallback = fallback
        self.cost = cost or default_cost
        self.seed_order = seed_order
        self.escalate = escalate
        # Resolved HERE, not as a default argument: a default argument is
        # evaluated at import and would freeze the module constant, breaking
        # both `factor_order_search.CONFIRM_EXTRA = ...` and any later retune.
        self.confirm_extra = (CONFIRM_EXTRA if confirm_extra is None
                              else max(0, int(confirm_extra)))
        self._seed = int(seed)
        self._rng = random.Random(seed)
        self._bnode = self._engine._bnode
        self._pairing = [list(row) for row in pairing]
        self._nodes = [tuple(int(x) for x in g) for g in node_charges]

    # ---- the lower bound, which is where both optimizations come from ------

    def minimum_factors(self) -> int:
        """A PROVABLE lower bound on the number of factors in any order.

        At cone degree 1 the accumulated product is empty — no factor of degree
        `≥ 1` reaches a degree-1 charge — so `_forced` sees `f = {}` there and
        returns `Ω = {0: −target}` at every node whose leading datum is nonzero,
        whatever the order.  Those factors are unavoidable, so no factorisation
        has fewer.  With the default BPS leading data (`−1` on every node) the
        bound is exactly `rank`.

        Two uses, and both need it to be a theorem rather than an observation:
        the search stops as soon as it attains the bound (nothing can beat it),
        and the acyclic short-circuit below is *accepted* by attaining it.
        """
        zero_product: dict = {}
        return sum(
            1 for k in self._engine.cone if sum(k) == 1
            and self._engine._forced(zero_product, self._engine.target(k))
        )

    def _confirms_beyond(self, spec: Sequence[Vec]) -> bool:
        """`∏ E_𝖖(X_{spec_i})` still equals `S` on a cone WIDER by `confirm_extra`.

        Strictly stronger than the stopping margin it replaced — a margin can be
        defeated by any factorisation that merely stops early, this only by one
        that agrees with `S` on a larger cone and then fails above it — but it is
        **not a certificate**, because the depth is finite and a counterexample
        can always live past it.  Measured: cycle6 at `D = 7`, seed 1, passes this
        at depth 1 (exact at 8) and is wrong at 9.

        So what the caller gets is the depth, not a promise: `Result.confirmed_to`
        records how far the equality was actually checked, and `is_spec` means
        "verified that far".  Raise `confirm_extra` to buy more.

        Costs one extra `S` build plus one Nahm-sum expansion at the wider cone,
        paid once per search on a candidate that has already passed every cheap
        screen.
        """
        wider = self.degree_cap + self.confirm_extra
        engine = BPSFactorSpectrum(self._pairing, self._nodes, wider,
                             leading_data=self._engine._leading, order="lex")
        engine.run()
        return self._rebuilds(spec, engine.spectrum_generator(), cone=wider)

    def _reference(self, cone: int | None = None) -> dict:
        """`S` at `cone` (default the cutoff), built by the engine in `lex` order."""
        cap = self.degree_cap if cone is None else cone
        engine = BPSFactorSpectrum(self._pairing, self._nodes, cap,
                             leading_data=self._engine._leading, order="lex")
        engine.run()
        return engine.spectrum_generator()

    def _rebuilds_content(self, placed: Sequence[Piece], target: dict,
                          cone: int | None = None) -> bool:
        """`∏_a A_{γ_a}(Ω_a) == target` in-cone, by the INDEPENDENT sum route.

        The `_rebuilds` of any content, not only a spin-0 spec.  Each placed
        piece is `m_s(γ)^{a_s}`, so its slice of `Ω` is `a_s·χ_s`, i.e.
        `{j: a_s for j in −2s … 2s step 2}`; `bps_factor_spectrum.nahm_generators`
        flattens that into `E_𝖖` factors and
        `nahm_local.general_nahm_habiro` sums over index tuples per charge.  No
        factor is ever multiplied by another, which is what keeps this
        independent of the accumulate-and-multiply engine the search runs on.

        This exists because the sparse fallback CARRIES SPIN and so could not be
        checked by the spec-only route — the module reported it with no
        independent confirmation at all.  That limit was never mathematical: it
        was one missing diagonal `n²` in the Nahm shift.
        """
        from nahm_local import general_nahm_habiro

        generators: list[tuple] = []
        for charge, two_s, amount in placed:
            slice_omega = {j: amount for j in range(-two_s, two_s + 1, 2)}
            generators.extend(
                nahm_generators(self._engine.charge(charge), slice_omega))
        if not generators:
            return not target

        # ⟨a, b⟩ = a·B·b on the LATTICE, the same pairing `Theory` hands
        # `S_from_spec`; the generators carry lattice charges, not node
        # coordinates.
        lattice = [g[0] for g in generators]
        pairing = self._pairing
        dim = len(pairing)
        kmat = [[sum(a[p] * pairing[p][r] * b[r]
                     for p in range(dim) for r in range(dim))
                 for b in lattice] for a in lattice]

        cap = self.degree_cap if cone is None else cone
        engine = (self._engine if cap == self.degree_cap
                  else BPSFactorSpectrum(self._pairing, self._nodes, cap,
                                   leading_data=self._engine._leading,
                                   order="lex"))
        for k in engine.cone:
            gamma = engine.charge(k)
            if general_nahm_habiro(gamma, generators, kmat) != target.get(gamma, H0):
                return False
        return True

    def _rebuilds(self, spec: Sequence[Vec], target: dict,
                  cone: int | None = None) -> bool:
        """`∏ E_𝖖(X_{spec_i}) == target` in-cone, by an INDEPENDENT route.

        `recursive_spectrum.Theory.S_from_spec` expands the ordered product as a
        Nahm sum and shares no code with the factor recursion, so this is a genuine
        check rather than a restatement — the same standard
        `extract_spec_from_quiver` holds itself to.  It is what turns "the
        content is spin-0" into "these are the `E_𝖖` factors of `S`, in this
        order", and it is the guard that caught the shortcut reading `Ω`'s dict
        order instead of the placement order.
        """
        from recursive_spectrum import Theory

        theory = Theory("order-search", self._pairing, self._nodes,
                        CONE=self.degree_cap if cone is None else cone)
        rebuilt = theory.S_from_spec(list(spec))
        return all(rebuilt.get(g, H0) == target.get(g, H0)
                   for g in set(rebuilt) | set(target) if theory.in_cone(g))

    def _strip_shortcut(self) -> Result | None:
        """The acyclic answer, taken directly rather than searched for.

        On an acyclic quiver the source/sink strip order is this repo's own
        **mantle theorem** (acyclic ⇒ topological product spec) and it places
        exactly one spin-0 factor per node — attaining `minimum_factors()`, hence
        optimal, hence nothing a search could improve on.  Searching there is
        pure waste and it is not small waste: measured 33.7 s at A6 to rediscover
        what the strip gives in milliseconds.

        **Verified, not assumed.**  The theorem is about the quiver; what is
        returned here is the engine's actual output, so the shortcut checks that
        output *is* spin-0, positive and of minimum length, and returns `None`
        (falling through to the full search) if it is not.  A theorem-backed
        fast path that silently returns a worse factorisation when its hypothesis
        is subtly unmet is exactly the failure mode worth spending five lines on.
        """
        if acyclic_node_order(self._pairing, self._nodes) is None:
            return None
        built = BPSFactorSpectrum(self._pairing, self._nodes, self.degree_cap,
                            leading_data=self._engine._leading, order="strip")
        built.run()
        pieces: list[Piece] = []
        # In the engine's own PLACEMENT order, which is `_pkey`, not the order
        # `omega`'s dict happens to be in.  `omega` is filled in cone order
        # (degree, then lexicographic) while the factors are placed by the key,
        # and the two differ whenever the strip's node order is not the identity
        # — at the pentagon they are exact reverses.  A spec is an ORDERED
        # product, so reading the dict order silently returns the factors of a
        # different element.
        for charge, omega in sorted(built.omega.items(),
                                    key=lambda item: built._pkey(item[0], 0)):
            decomposed = spin_decompose(omega)
            if set(decomposed) != {0} or decomposed[0] < 1:
                return None
            pieces.append((charge, 0, decomposed[0]))
        if len(pieces) != self.minimum_factors():
            return None
        spec = [built.charge(k) for k, _s, amount in pieces
                for _ in range(amount)]
        if not self._rebuilds(spec, built.spectrum_generator()):
            return None
        if not self._confirms_beyond(spec):
            return None
        content = Content(factors=len(built.omega), terms=sum(p[2] for p in pieces),
                          spin_weight=0, max_spin_doubled=0, negatives=0,
                          forced_next=0)
        return Result(
            order=[built.charge(k) for k, _s, _a in pieces],
            omega={built.charge(k): dict(om) for k, om in built.omega.items()},
            spectrum_generator=built.spectrum_generator(),
            content=content,
            is_spec=True,
            confirmed_to=self.degree_cap + self.confirm_extra,
            top_degree=max(sum(k) for k, _s, _a in pieces),
            margin=self.degree_cap - max(sum(k) for k, _s, _a in pieces),
            stopped=(max(sum(k) for k, _s, _a in pieces) + STOP_MARGIN
                     <= self.degree_cap),
            spec=spec,
            exhaustive=True,
            optimal=True,
            route="strip",
            _pieces=pieces,
        )

    # ---- helpers ----------------------------------------------------------

    def _commutes(self, a: Vec, b: Vec) -> bool:
        """`⟨a, b⟩ == 0` in node coordinates — the trace-monoid independence."""
        return sum(a[i] * self._bnode[i][j] * b[j]
                   for i in range(self.rank) for j in range(self.rank)) == 0

    def _fresh(self, state: _State, degree: int):
        """The forced `Ω` at this degree, or `None` if the branch is cut.

        `None` is a decision and not a preference: the returned `Ω` is *forced*
        by the strictly-lower-degree factors, so a spin or a negative
        multiplicity appearing here is permanent.

        Reuses the memo `_content` left behind: scoring a state one layer earlier
        already computed exactly this readout, and it is the dominant cost.
        """
        if state.lookahead is not None and state.lookahead[0] == degree:
            fresh = state.lookahead[1]
        else:
            fresh = self._engine._readout(state.product, degree)
        for _charge, omega in fresh:
            for two_s, amount in spin_decompose(omega).items():
                if self.require_spin_zero and two_s != 0:
                    return None
                if self.require_positive and amount < 0:
                    return None
        return fresh

    def _insert(self, state: _State, position: int, piece: Piece) -> _State:
        """A copy of `state` with `piece` placed at `position`.

        Prefixes before the insertion point are shared, not copied: they are the
        product of the same factors in the same order, so they stay valid.  Only
        the suffix is rebuilt, which is `bps_factor_spectrum._run_insert`'s own
        invariant applied per branch.
        """
        placed = state.placed[:position] + [piece] + state.placed[position:]
        prefix = state.prefix[:position + 1]
        prefix_deg = state.prefix_deg[:position + 1]
        for i in range(position, len(placed)):
            k0, two_s, amount = placed[i]
            spread = {j: amount for j in range(-two_s, two_s + 1, 2)}
            nxt, nxt_deg = self._engine._charge_multiply(
                prefix[i], prefix_deg[i], k0, sum(k0), spread)
            prefix.append(nxt)
            prefix_deg.append(nxt_deg)
        return _State(placed, prefix, prefix_deg, dict(state.omega))

    def _placement(self, state: _State, positions, pieces) -> list:
        """The child's factor list — integer work only, no arithmetic."""
        placed = list(state.placed)
        for position, piece in zip(positions, pieces):
            placed.insert(position, piece)
        return placed

    def _rank_and_build(self, candidates, degree, last_layer):
        """Score every candidate cheaply, then pay full price for the survivors.

        THE OPTIMIZATION THIS FUNCTION EXISTS FOR.  Ranking needs one number the
        candidate does not already know — the terms forced at degree `d+1` — and
        that readout only ever looks at charges of degree `d+1`.  Building the
        full-cone product to obtain it is waste, and it was most of the runtime:
        with `keep_per_degree` states each branching up to `branch_cap` ways, the
        search was building a few hundred products per layer in order to keep a
        handful.

        So the product is built twice, cheaply then properly: once truncated to
        `d+1` for the score, and again at the full cone only for the states that
        survive the cut.  The truncated build is the same arithmetic on a much
        smaller cone — at rank 6 with `D = 7`, scoring at degree 2 touches 28 cone
        points against 1716 — and the parent's prefix can simply be filtered down
        to that cone first, since truncating a product is free.
        """
        scored = []
        for state, positions, pieces, fresh, placed in candidates:
            omega = dict(state.omega)
            for charge, forced in fresh:
                omega[charge] = forced
            forced_next = self._forced_cheaply(state, positions, pieces,
                                               degree + 1)
            scored.append((self.cost(self._summarise(omega, forced_next)),
                           state, positions, pieces, omega))
        scored.sort(key=lambda row: row[0])
        keep = (scored if self.keep_per_degree is None
                else scored[:self.keep_per_degree])
        built = []
        for _cost, state, positions, pieces, omega in keep:
            child = self._insert_many(state, positions, pieces,
                                      build=not last_layer)
            child.omega = omega
            child.content = self._content(child, degree + 1)
            built.append(child)
        self._trimmed = self._trimmed or len(keep) < len(scored)
        return built

    def _forced_cheaply(self, state: _State, positions, pieces,
                        next_degree: int):
        """Weighted terms forced at `next_degree`, on a cone truncated to it.

        Everything above `next_degree` is irrelevant to this readout, so it is
        not computed.  Returns 0 past the cutoff, where there is nothing to read.

        ⚠ **An extra prune here was tried and MEASURED WORSE — do not re-add it.**
        The `Ω` read here is *forced*, so a spin in it condemns the candidate
        exactly as it would one layer later; cutting on the spot looks free, since
        the readout is already done.  It is not: measured 8 % *slower* over the
        rank-4 dictionary (7.1 s → 7.7 s) with identical results, and it rescued
        no answer at any narrower width.  The reason is worth keeping — the cost
        function **already subsumes it**.  `SPIN_PENALTY` weights exactly this
        forced content, so a doomed candidate already sorts last and is already
        dropped by the width; the explicit prune re-derives a decision the
        ranking has made, and pays for the bookkeeping.  That it is redundant is
        evidence the cost is shaped right.
        """
        if next_degree > self.degree_cap:
            return 0
        placed = list(state.placed)
        first = len(placed)
        for position, piece in zip(positions, pieces):
            placed.insert(position, piece)
            first = min(first, position)

        engine = self._engine
        full_cap = engine.degree_cap
        engine.degree_cap = next_degree
        try:
            degrees = state.prefix_deg[first]
            acc = {k: v for k, v in state.prefix[first].items()
                   if degrees[k] <= next_degree}
            acc_deg = {k: d for k, d in degrees.items() if d <= next_degree}
            for i in range(first, len(placed)):
                charge, two_s, amount = placed[i]
                spread = {j: amount for j in range(-two_s, two_s + 1, 2)}
                acc, acc_deg = engine._charge_multiply(acc, acc_deg, charge,
                                                    sum(charge), spread)
            total = 0
            for _charge, forced in engine._readout(acc, next_degree):
                for two_s, amount in spin_decompose(forced).items():
                    total += weighted_terms(abs(amount), abs(amount) * two_s)
            return total
        finally:
            engine.degree_cap = full_cap

    @staticmethod
    def _summarise(omega: dict, forced_next: int) -> Content:
        """`Content` from an already-known `Ω` and a precomputed forced count."""
        spins = terms = spin_weight = negatives = 0
        for content in omega.values():
            for two_s, amount in spin_decompose(content).items():
                spins = max(spins, two_s)
                terms += abs(amount)
                spin_weight += abs(amount) * two_s
                negatives += amount < 0
        return Content(factors=len(omega), terms=terms, spin_weight=spin_weight,
                       max_spin_doubled=spins, negatives=negatives,
                       forced_next=forced_next)

    def _materialise(self, state: _State) -> _State:
        """Finish a deferred state's arithmetic.  Idempotent."""
        if state.pending is None:
            return state
        first = state.pending
        prefix, prefix_deg = state.prefix, state.prefix_deg
        for i in range(first, len(state.placed)):
            charge, two_s, amount = state.placed[i]
            spread = {j: amount for j in range(-two_s, two_s + 1, 2)}
            nxt, nxt_deg = self._engine._charge_multiply(
                prefix[i], prefix_deg[i], charge, sum(charge), spread)
            prefix.append(nxt)
            prefix_deg.append(nxt_deg)
        state.pending = None
        return state

    def _insert_many(self, state: _State, positions: Sequence[int],
                     pieces: Sequence[Piece], *, build: bool = True) -> _State:
        """All of a degree's pieces placed at once, rebuilding the suffix ONCE.

        The obvious way — call `_insert` per piece — rebuilds an *overlapping*
        suffix each time, so placing `m` pieces costs `m` passes over roughly the
        same tail.  Profiling says that is the whole game: `_insert` was 95 % of
        runtime, and inside it 3301 accumulator products for 941 inserts, all of
        it exact Habiro arithmetic.  Doing the list surgery first (integer work)
        and the arithmetic once collapses those passes into one.

        The rebuild point is `min(positions)` taken on the *raw* positions, which
        is a safe lower bound for the earliest invalidated prefix: a later
        insertion at a smaller index only shifts an earlier one further right, so
        no prefix below the minimum can have changed.  Same invariant
        `bps_factor_spectrum._run_piece_insert` uses, applied per branch.
        """
        placed = list(state.placed)
        first = len(placed)
        for position, piece in zip(positions, pieces):
            placed.insert(position, piece)
            first = min(first, position)
        prefix = state.prefix[:first + 1]
        prefix_deg = state.prefix_deg[:first + 1]
        child = _State(placed, prefix, prefix_deg, dict(state.omega))
        if not build:
            # DEFERRED.  At the last cone degree the ranking needs no product at
            # all: `_content`'s lookahead is empty past the cutoff, and the term
            # and spin counts come from `Ω`, which is already known.  Only the
            # winner's product is ever read — by `_rebuilds` and the report.
            #
            # ⚠ MEASURED NEUTRAL on a search that SUCCEEDS, and kept anyway.
            # The reasoning above is sound but nearly vacuous where it matters
            # most: a search that finds a factorisation has stopped populating
            # well below the cutoff (that is what `STOP_MARGIN` demands), so the
            # top layers place nothing and there is nothing to defer.  Measured
            # 24.08 s -> 24.02 s on cycle6 at cutoff 7.  It does bite on the
            # dense cases — where factors keep appearing to the boundary, which
            # is exactly when the accumulator is longest — and it costs nothing,
            # so it stays.  Recorded as neutral rather than banked as a win.
            child.pending = first
            return child
        for i in range(first, len(placed)):
            charge, two_s, amount = placed[i]
            spread = {j: amount for j in range(-two_s, two_s + 1, 2)}
            nxt, nxt_deg = self._engine._charge_multiply(
                prefix[i], prefix_deg[i], charge, sum(charge), spread)
            prefix.append(nxt)
            prefix_deg.append(nxt_deg)
        return child

    def _positions(self, placed_count: int, piece_count: int):
        """Position tuples to try, deterministic first and then sampled.

        Enumerating every tuple is `(n+1)(n+2)…(n+m)` and hopeless past the
        smallest cases, so the cap is real.  Taking the deterministic ones first
        means an uncapped run and the head of a capped run agree, and the sampled
        tail is seeded, so a capped run is reproducible rather than merely
        bounded.
        """
        import itertools

        # LATEST POSITION FIRST.  Appending is the one insertion that rebuilds no
        # suffix — `_insert` at the end reuses every cached prefix — so a capped
        # run explores its cheapest placements first, and the degree-compatible
        # orders (which append by construction, and are the ones `"lex"` and
        # `"degree-key"` realise) are the ones a cap keeps rather than the ones
        # it drops.  Uncapped this is only an enumeration order and changes
        # nothing: the layer is sorted by cost afterwards either way.
        ranges = [list(reversed(range(placed_count + offset + 1)))
                  for offset in range(piece_count)]
        total = 1
        for r in ranges:
            total *= len(r)
        if total <= self.branch_cap:
            return list(itertools.product(*ranges)), True
        out, seen = [], set()
        for tup in itertools.product(*ranges):
            out.append(tup)
            seen.add(tup)
            if len(out) >= self.branch_cap // 2:
                break
        while len(out) < self.branch_cap:
            tup = tuple(self._rng.randrange(len(r)) for r in ranges)
            if tup not in seen:
                seen.add(tup)
                out.append(tup)
        return out, False

    def _content(self, state: _State, next_degree: int) -> Content:
        """Summarise a state: what it has placed, and what it has already forced.

        `forced_next` is the weighted term count the candidate has committed
        itself to at the degree above — forced by what is placed, so unchangeable.
        It is counted alongside the placed terms rather than beside them, because
        it is the same kind of thing (see `default_cost`), and because every
        placement of a degree's pieces gives THAT degree the same `Ω` — so a
        count of placed terms alone is constant across exactly the sibling set it
        is being asked to rank.
        """
        spins = terms = spin_weight = negatives = 0
        for omega in state.omega.values():
            for two_s, amount in spin_decompose(omega).items():
                spins = max(spins, two_s)
                terms += abs(amount)
                spin_weight += abs(amount) * two_s
                negatives += amount < 0
        ahead = 0
        if next_degree <= self.degree_cap:
            fresh = self._engine._readout(state.product, next_degree)
            state.lookahead = (next_degree, fresh)
            for _charge, omega in fresh:
                for two_s, amount in spin_decompose(omega).items():
                    ahead += weighted_terms(abs(amount), abs(amount) * two_s)
        return Content(factors=len(state.omega), terms=terms,
                       spin_weight=spin_weight, max_spin_doubled=spins,
                       negatives=negatives, forced_next=ahead)

    # ---- the loop ---------------------------------------------------------

    def search(self) -> Result:
        """Run the BFS and return the best factorisation found.

        Takes the acyclic short-circuit first where it applies and verifies, so
        the BFS runs only on the quivers that actually need it — which are
        exactly the ones whose source/sink strip leaves a core, i.e. where the
        engine's default order has nothing to say and falls back to random
        placement.
        """
        shortcut = self._strip_shortcut()
        if shortcut is not None:
            return shortcut

        found = self._run(self.seed_order)
        if found.spec is None and self.seed_order is not None:
            # The seed was worse than nothing here — its clean prefix led into a
            # dead end the from-scratch beam avoids.  Retry without it rather
            # than report a failure caused by the caller's hint.
            found = self._run()
        if found.spec is None and self.escalate and self.keep_per_degree:
            found = self._escalate() or found
        # Fall back only when the spin-0 pass found NO factorisation at all.
        # Deliberately keyed on `spec is None`, not on `is_spec`: since
        # `STOP_MARGIN` landed, `is_spec` is also False when a perfectly good
        # spin-0 factorisation was found but stops too close to the cutoff to be
        # certified.  Falling back there would throw away the candidate and
        # return an unpruned answer instead of the right advice, which is to
        # raise the cutoff — measured doing exactly that on the rank-5, -6 and
        # -8 cycles at cutoff 5.
        if (found.spec is not None or not self.require_spin_zero
                or not self.fallback):
            return found

        # NO SPEC EXISTS HERE (or none was reachable), and the spin-0 prune
        # therefore threw away every candidate.  A sparse `Ω` is worth having on
        # its own — *"a sparsely populated S helps with calculations too, so it
        # is a good optimization goal"* — so rather than
        # returning nothing, drop the prune and let the cost do the work.  The
        # spin penalty still pushes towards low spin; what is given up is only
        # the guarantee, which was already unattainable on this quiver.
        relaxed = FactorOrderSearch.__new__(FactorOrderSearch)
        relaxed.__dict__.update(self.__dict__)
        relaxed.require_spin_zero = False
        relaxed.fallback = False
        sparse = relaxed._run()
        sparse.route = "search-sparse"
        if not sparse.order:
            return found
        # VERIFIED, since 2026-08-14 it can be.  This route carries spin, so the
        # spec-only rebuild could not touch it and the module reported it with no
        # independent check at all — the one output whose correctness rested on
        # the search's own arithmetic.  `_rebuilds_content` closes that.
        sparse.content_verified = self._rebuilds_content(
            list(sparse._pieces), self._reference())
        return sparse

    def _escalate(self) -> Result | None:
        """RESTART on a failure — different samples, same width.  Measured 2026-08-14.

        A negative result from this search is "not found", not "does not exist",
        and the measurement behind this method is *how much* of that gap is mere
        sampling.  `_positions` samples once the position count exceeds its cap,
        so a run is a draw, and the draws differ wildly.  At the **default** width
        on two rank-5 dictionary entries, varying only the seed::

            linear_A1_U3_Nf1[mut+1]      seeds 1-8:  - - - - - -  8  8
            linear_A1_U3_Nf1[mut+1]x6    seeds 1-8:  - -  9 - -  9  9  9

        Both answers are reachable at the width that first reported nothing.  So
        restarting is the mechanism, and it is worth separating from two things
        it is easy to confuse with:

        * **Not widening.**  A wider beam *also* finds these, but only because
          changing the width perturbs the same sampling — at fixed width 16, seeds
          1/2/3 give no-spec / 10 factors / 9 factors.  Widening was the first
          thing tried here and its apparent benefit was a sampling artifact,
          caught only when two runs of the same quantity disagreed.
        * **Not a fitted constant.**  Widths are not ordered by quality, so no
          width is "the right one" — which is the shape of error that produced
          `STOP_MARGIN` twice.  A restart budget *is* monotone: more restarts can
          only find more, never less.  That makes `RESTARTS` an honest cost dial
          rather than a calibrated threshold, and it is why this ladder restarts
          before it widens.

        Safety and cost.  Every attempt passes the same `CONFIRM_EXTRA`
        certificate, so a retry can only find a right answer, never introduce a
        wrong one.  It fires only when the first pass returned no spec at all —
        8 of 305 rank-5 dictionary entries — so it is free on success.  Restarts
        cost about what the original pass cost (~1 s here); the final unbounded
        widening is kept as a last resort but is the expensive rung (measured 300 s
        on one entry), so it runs last and only if every restart failed.

        Returns the first certified result, or `None`, in which case the caller
        keeps the original answer and the sparse fallback proceeds as before.
        """
        def attempt(width, seed):
            retry = FactorOrderSearch.__new__(FactorOrderSearch)
            retry.__dict__.update(self.__dict__)
            retry.keep_per_degree = width
            retry.escalate = False        # one ladder, not a ladder per rung
            retry._rng = random.Random(seed)
            out = retry._run()
            return out if out.spec is not None else None

        base = self.keep_per_degree
        for step in range(1, RESTARTS + 1):
            out = attempt(base, self._seed + step)
            if out is not None:
                out.route = "search-retry"
                return out
        for width in (base * 4, None):
            out = attempt(width, self._seed)
            if out is not None:
                out.route = "search-wide"
                return out
        return None

    def _replay(self, seed_key: Callable[[Vec, int], object]):
        """Replay `seed_key` as far as it stays clean.  Returns `(state, degree)`.

        The seed is a `piece_key` — a total order on the pairs `(γ, s)`, i.e.
        exactly what a previous search returns from `Result.piece_key()`.  Each
        degree's forced pieces are inserted so that the placed list stays sorted
        by that key, which reproduces the seed's placement *including its
        interleaving across degrees*: the previous run placed in this order, so
        re-inserting in it lands every piece where it was.

        The return is the state at the end of the last degree that passed the
        prunes, and the degree the caller should resume the BFS at.  So a seed
        that survives to the top costs one build and no search; a seed that dies
        at degree `d` still donates degrees `1 … d−1`, which is where the beam
        would otherwise be carrying `keep_per_degree` candidates apiece.

        The prefix is donated, NOT trusted: whatever the seed contributes is
        re-verified by the same `_rebuilds` / `_confirms_beyond` the from-scratch
        route uses, so a stale seed can cost time but cannot corrupt an answer.
        """
        zero = tuple([0] * self.rank)
        state = _State([], [{zero: H1}], [{zero: 0}], {})
        keys: list = []
        resume = 1
        for degree in range(1, self.degree_cap + 1):
            fresh = self._fresh(state, degree)
            if fresh is None:
                return state, degree          # the seed dies here; BFS takes over
            resume = degree + 1
            if not fresh:
                continue
            pieces = [(charge, two_s, amount)
                      for charge, omega in fresh
                      for two_s, amount in sorted(spin_decompose(omega).items())]
            omega = dict(state.omega)
            for charge, forced in fresh:
                omega[charge] = forced
            for piece in sorted(pieces, key=lambda p: seed_key(p[0], p[1])):
                key = seed_key(piece[0], piece[1])
                position = bisect.bisect_right(keys, key)
                state = self._insert(state, position, piece)
                keys.insert(position, key)
            state.omega = omega
        return state, resume

    def _run(self, seed_key: Callable[[Vec, int], object] | None = None) -> Result:
        """One pass of the BFS, under the current prunes."""
        self._trimmed = False
        floor = self.minimum_factors()
        zero = tuple([0] * self.rank)
        states = [_State([], [{zero: H1}], [{zero: 0}], {})]
        exhaustive = True
        examined = cut = 0
        start = 1
        if seed_key is not None:
            seeded, start = self._replay(seed_key)
            states = [seeded]
            if start > self.degree_cap:
                self._materialise(seeded)
                return self._report(seeded, exhaustive, examined, cut, floor)

        for degree in range(start, self.degree_cap + 1):
            layer: list[_State] = []
            seen: set = set()
            candidates: list = []
            last_layer = degree == self.degree_cap
            for state in states:
                fresh = self._fresh(state, degree)
                if fresh is None:
                    cut += 1
                    continue
                if not fresh:
                    layer.append(state)
                    continue
                pieces = [(charge, two_s, amount)
                          for charge, omega in fresh
                          for two_s, amount in sorted(spin_decompose(omega).items())]
                positions, complete = self._positions(len(state.placed),
                                                      len(pieces))
                exhaustive = exhaustive and complete
                last_layer = degree == self.degree_cap
                for tup in positions:
                    placed = self._placement(state, tup, pieces)
                    key = _normal_form(placed, self._commutes)
                    if key in seen:
                        continue
                    seen.add(key)
                    examined += 1
                    candidates.append((state, tup, pieces, fresh, placed))
            if candidates:
                layer.extend(self._rank_and_build(candidates, degree,
                                                  last_layer))
            if not layer:
                return Result(exhaustive=exhaustive, states_examined=examined,
                              branches_cut=cut)
            # NOT narrowed to a state that has attained the floor here.  Doing
            # that was tried and is UNSOUND: `len(placed) == floor` at an
            # intermediate degree says only that no *extra* factor has appeared
            # YET, and further factors can still be forced above — so pruning the
            # alternatives away loses the orders that stay clean to the top.
            # Measured: it silently turned the SU(3)-cyclic 6-factor spec into
            # "no spin-0 order".  The floor is a REPORTING fact (`optimal`) and a
            # justification for the acyclic shortcut, not a mid-search prune.
            # The cut already happened in `_rank_and_build`, on the cheap
            # scores; states that passed through untouched (no fresh factors at
            # this degree) are simply carried.
            layer.sort(key=lambda st: self.cost(st.content)
                       if st.content is not None else ())
            if self.keep_per_degree is not None and len(layer) > self.keep_per_degree:
                layer = layer[:self.keep_per_degree]
                self._trimmed = True
            exhaustive = exhaustive and not self._trimmed
            states = layer

        best = min(states, key=lambda st: self.cost(
            self._content(st, self.degree_cap + 1)))
        self._materialise(best)
        return self._report(best, exhaustive, examined, cut, floor)

    def _report(self, state: _State, exhaustive: bool, examined: int,
                cut: int, floor: int) -> Result:
        content = self._content(state, self.degree_cap + 1)
        spec = None
        if content.is_spin_zero and content.negatives == 0:
            spec = []
            for charge, _two_s, amount in state.placed:
                spec.extend([self._engine.charge(charge)] * amount)
            # Same independent rebuild the shortcut is held to.  A spin-0
            # content is a *claim* that `S` factors as these `E_𝖖`s in this
            # order; the claim is checked before it is reported, so `is_spec`
            # never rests on the search's own arithmetic alone.
            if not self._rebuilds(spec, state.product):
                spec = None
        top_degree = max((sum(k) for k, _s, _a in state.placed), default=0)
        stopped = (bool(state.placed)
                   and top_degree + STOP_MARGIN <= self.degree_cap)
        # `stopped` is only the cheap screen.  The certificate is the rebuild on
        # a strictly WIDER cone, because a factorisation can stop well inside the
        # cutoff, rebuild perfectly there, and still disagree one degree up —
        # measured at cycle6, which is what retired the margin as a certificate.
        confirmed = bool(spec) and stopped and self._confirms_beyond(spec)
        return Result(
            order=[self._engine.charge(k) for k, _s, _a in state.placed],
            omega={self._engine.charge(k): dict(om)
                   for k, om in state.omega.items()},
            spectrum_generator={self._engine.charge(k): c
                                for k, c in state.product.items()},
            content=content,
            is_spec=confirmed,
            confirmed_to=(self.degree_cap + self.confirm_extra) if confirmed else 0,
            top_degree=top_degree,
            margin=self.degree_cap - top_degree,
            stopped=stopped,
            spec=spec,
            exhaustive=exhaustive,
            optimal=confirmed and len(state.placed) == floor,
            route="search",
            states_examined=examined,
            branches_cut=cut,
            _pieces=list(state.placed),
        )


def find_simple_factorisation(pairing, node_charges, cutoff, **kwargs) -> Result:
    """Search for a spin-0 factorisation of `S` — the module's front door.

    Returns a `Result`; `result.is_spec` says whether the content is spin-0 only,
    `result.spec` gives the `E_𝖖` factors as a charge list in placement order,
    and `result.piece_key()` hands the winning order back to
    `bps_factor_spectrum.BPSFactorSpectrum` so `S` can be rebuilt in it.
    """
    return FactorOrderSearch(pairing, node_charges, cutoff, **kwargs).search()


def simplify_factorisation(spectrum, pairing, node_charges, cutoff,
                           *, verify: bool = True, **kwargs) -> Result:
    """Simplest factorisation of an `S` you ALREADY HAVE.

    > **The author, 2026-08-13.**  *"We should also look for an algorithm to find the
    > simplest factorization of an already-computed `S`.  Simplifying could be
    > useful even at intermediate stages, before pushing a cutoff higher."*

    Same search as `find_simple_factorisation`, entered from the other end: the
    caller supplies `S` and it supplies its own leading data
    (`leading_data_from_spectrum`), so nothing about where it came from — which
    order built it, which engine, whether it was read out of the dictionary —
    needs to be known or re-derived.

    `verify=True` (the default) is a **positive control, not a formality**: it
    rebuilds `S` from the derived leading data and refuses to factor anything
    that does not match in-cone.  Without it the function would silently return a
    perfectly valid factorisation *of a different element* whenever the supplied
    `S` is not this quiver's — which is precisely the mistake a re-factorisation
    tool invites, since its whole job is to take an `S` on trust.

    THE INTERMEDIATE-STAGE WORKFLOW the author names is what `seed_order` is for.
    Simplify at the cutoff you have, then carry the order up::

        low  = simplify_factorisation(S, pairing, nodes, 5)
        high = find_simple_factorisation(pairing, nodes, 6,
                                         seed_order=low.piece_key())

    and the seed is donated rather than trusted — the second call re-verifies
    from scratch, so a seed that no longer works at the higher cutoff costs time
    and nothing else.  That guard is load-bearing: replaying a good low-cutoff
    order verbatim at a higher one was **measured to fail 4 times in 5**, so the
    hint has to be re-checked, and `_replay` stops donating at the first degree
    where the seed's forced `Ω` goes bad.
    """
    leading = leading_data_from_spectrum(spectrum)
    if verify:
        control = BPSFactorSpectrum(pairing, node_charges, cutoff,
                              leading_data=leading, order="lex")
        control.run()
        rebuilt = control.spectrum_generator()
        charges = set(rebuilt) | {tuple(g) for g in spectrum}
        bad = [g for g in charges
               if rebuilt.get(g, H0) != spectrum.get(g, H0)]
        if bad:
            raise ValueError(
                f"the supplied S is not this quiver's crystalline S at cutoff "
                f"{cutoff}: {len(bad)} charge(s) differ from the rebuild, e.g. "
                f"{sorted(bad)[0]}.  Factoring it would return a valid "
                f"factorisation of a DIFFERENT element.  Pass verify=False only "
                f"if that is deliberate.")
    kwargs.setdefault("leading_data", leading)
    return FactorOrderSearch(pairing, node_charges, cutoff, **kwargs).search()


def _main() -> None:
    import time

    def three_cycle(a, b, c):
        return [[0, a, -c], [-a, 0, b], [c, -b, 0]]

    b2 = [(1, 0), (0, 1)]
    b3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    b4 = [tuple(1 if i == k else 0 for i in range(4)) for k in range(4)]
    cases = (
        ("pentagon", [[0, 1], [-1, 0]], b2, 6),
        ("pure SU(2)", [[0, 2], [-2, 0]], b2, 6),
        ("3-cycle(1,1,1)", three_cycle(1, 1, 1), b3, 5),
        ("SU(3) cyclic", [[0, 1, 0, -2], [-1, 0, 2, 0], [0, -2, 0, 1],
                          [2, 0, -1, 0]], b4, 4),
        ("Markov(2,2,2)", three_cycle(2, 2, 2), b3, 4),
    )
    for name, pairing, nodes, degree in cases:
        started = time.time()
        found = find_simple_factorisation(pairing, nodes, degree)
        elapsed = time.time() - started
        verdict = ("spec " + str(found.spec)) if found.is_spec else "no spin-0 order"
        print(f"{name:16s} [{found.route}] {verdict}")
        print(f"{'':16s} optimal={found.optimal} exhaustive={found.exhaustive} "
              f"states={found.states_examined} cut={found.branches_cut} "
              f"{elapsed:.2f}s")


if __name__ == "__main__":
    _main()
