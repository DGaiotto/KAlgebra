"""`graded_rg_solver` — the generalised F-solver over a graded IR (Plan 20, T2).

Solves the **discovery relation** in an arbitrary graded auxiliary
(IR) `KAlgebra`:

    RG(a) · S_RG  =  L_apex  +  O(q)

for the unknown `RG(a)` (an auxiliary `Element` with bar-invariant /
palindromic `Z[q^±]` coefficients), given the spectrum generator `S_RG`
and the apex target label `apex` (the abstract lower-tropical label,
the `X_γ` analogue).

This is `bps_kalgebra_internals._solve_F_by_peeling` lifted off the
quantum torus.  The per-label scalar peel is unchanged — the
bar-invariant `[n]_q` cancellation of the non-positive-`q` residual —
but:

* labels carry charges in the grading lattice `Γ_RG` and are walked in
  **increasing height** (the abstract positive-cone order), and
* propagation `f_c · [S_RG]_s` uses the **IR's own `multiply`** (general
  structure constants) rather than the quantum-torus `q`-twist, so a
  charge `p` may carry a whole multi-dimensional `B_p` (several aux
  labels), handled label-by-label.

Two facts make this correct and terminating (per the design session):

1. **`[S_RG]_0 = 1_B`** (degree-0 part of `S_RG` is exactly the
   identity): the unknown `[RG(a)]` at a label `c` enters
   `[RG(a)·S_RG]_c` through the self-term `f_c · 1_B = f_c`, read off
   directly — the solve is unit-triangular, no inversion.

2. **Exact Nahm/Habiro arithmetic.**  `S_RG` coefficients are exact
   `HabiroElement`s (e.g. `rg_generator(cutoff)`), and the residual is
   accumulated exactly — *no `q`-truncation*.  So all cancellations
   happen before the `O(q)` check: beyond the (finite) support the
   residual is genuinely `O(q)` (empty non-positive part ⇒ `f = 0` ⇒ no
   onward propagation).  Combined with the finite charge-support of
   `rg_generator(cutoff)`, the reachable set is finite and **the walk
   stops on its own** — termination is recognised as "no label left
   with a non-positive-`q` residual" (empty worklist).  No explicit
   upper-tropical bound is needed.

(Truncating coefficients to `q^≤K` — the first draft's bug — destroys
the cancellations, so beyond-support residuals spuriously look
non-`O(q)` and the walk diverges.  Hence: exact arithmetic only.)

`solve_rg` returns `(RG(a), ρ_UV⁻¹(a))` — the latter from the mirror
constraint.  Forward `ρ_UV(a)` and the **alternative RG map** `tRG(a)`
(defined by `RG(a)·S_RG = S_RG·tRG(a)`) come from `solve_trg`, which is
just `solve_rg` in the **opposite algebra** (the left discovery relation
`S_RG·tRG = a + O(q)`); `ρ_UV` and `ρ_IR` intertwine `RG` and `tRG`
(`RG ∘ ρ_UV = ρ_IR ∘ tRG`).
"""

from __future__ import annotations

import heapq

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element, KAlgebra
from laurent_poly import LaurentPoly
from habiro import HabiroElement
from grading import Grading

Label = tuple


# ---------------------------------------------------------------------------
# Bar-invariant non-positive peel (the [n]_q cancellation, as in BPS)
# ---------------------------------------------------------------------------


def _peel_nonpositive(nonpos: dict[int, int]) -> dict[int, int]:
    """Given the non-positive-`q` part `nonpos` (`{q-power ≤ 0: coeff}`)
    of a residual, return the LaurentPoly coeff-dict of the *bar-invariant*
    correction `f` whose non-positive part is `−nonpos`.

    Identical to the `[n]_q` peel in
    `bps_kalgebra_internals._solve_F_by_peeling`: cancel the most-negative
    term `q^k` (coeff `c`) with `−c·[n]_q`, `n = 1−k`; the chosen `[n]_q`
    contributes at `k, k+2, …, −k`, whose own non-positive tail folds back
    into `nonpos`.  Output is automatically palindromic (bar-invariant).
    """
    nonpos = dict(nonpos)
    f_nq: dict[int, int] = {}        # [n]_q basis: n -> coeff
    while nonpos:
        k = min(nonpos)
        c = nonpos.pop(k)
        if c == 0:
            continue
        n = 1 - k
        f_nq[n] = f_nq.get(n, 0) - c
        if f_nq[n] == 0:
            del f_nq[n]
        for i in range(1, n):
            exp = k + 2 * i
            if exp > 0:
                break
            nonpos[exp] = nonpos.get(exp, 0) - c
            if nonpos[exp] == 0:
                del nonpos[exp]
    # Expand the [n]_q correction to a LaurentPoly coeff dict.
    f_lp: dict[int, int] = {}
    for nq, cn in f_nq.items():
        e = -(nq - 1)
        while e <= nq - 1:
            f_lp[e] = f_lp.get(e, 0) + cn
            e += 2
    return {e: c for e, c in f_lp.items() if c != 0}


def _leading_q(h: HabiroElement) -> int | None:
    """Leading (lowest) `q`-exponent of an exact `HabiroElement`.

    Each denominator factor `(1−q^{2k})` has constant term 1, so the
    Laurent-expansion leading exponent equals the numerator's lowest
    exponent.  Returns `None` for zero.
    """
    if h.numerator.is_zero():
        return None
    return min(h.numerator._coeffs)


# ---------------------------------------------------------------------------
# The discovery solve
# ---------------------------------------------------------------------------


class _MirrorAcc:
    """Incremental evaluator of the **upper (mirror) constraint** on the
    growing candidate `RG(a) = f`:

        ρ_IR(S_RG) · RG(a)  =  L_upper  +  O(q).

    Maintains the exact Habiro product `acc = ρ_IR(S_RG)·f` and updates it
    by **only the newly-added `f`-term** (`add`); `test` returns the
    `upper` label iff `RG(a)` is *complete*, else `None`.  The incremental
    `acc` is identical to a full recompute (each `f`-entry is set once),
    and avoids the O(|f|²) cost of recomputing the whole product per peel.

    Completion certificate — **cutoff stability** (the robust replacement
    for the old "isolated by a height gap" heuristic, which gave both
    false positives (premature close → truncated `RG`) and false negatives
    (never close → infinite cone-climb)).  `S_RG` is charge-truncated, so
    `ρ_IR(S_RG)·f` carries deep-edge `q⁰` **artifacts whose location moves
    with the truncation cutoff**.  The genuine non-positive structure is
    cutoff-*independent*:

    * a **complete** `f` leaves exactly **one** stable non-positive entry,
      the single upper element `{0:1}` (`= L_upper`);
    * a **premature** `f` leaves a stable non-positive **residual chain**
      (more than one stable entry, or a non-`{0:1}` one).

    So we evaluate the mirror at **two cutoffs** (the given `S_RG` and a
    higher-cutoff `S_RG`); an entry is *genuine* iff its non-positive part
    is identical in both, and `f` is complete iff the genuine set is
    exactly one `{0:1}` entry (whose label is `upper = ρ_UV⁻¹(a)`).  The
    deep-edge artifacts differ between the two cutoffs and are filtered
    out.  (Verified against the `SubquiverRG` oracle: premature `f` shows
    a multi-entry stable chain, complete `f` shows the single stable top.)
    """

    __slots__ = ("aux", "rho_s_lo", "rho_s_hi", "grading", "acc_lo", "acc_hi")

    def __init__(self, aux: KAlgebra,
                 rho_s_lo: dict[Label, HabiroElement],
                 rho_s_hi: dict[Label, HabiroElement] | None,
                 grading: Grading) -> None:
        self.aux = aux
        self.rho_s_lo = rho_s_lo
        self.rho_s_hi = rho_s_hi            # None ⇒ single-cutoff gap fallback
        self.grading = grading
        self.acc_lo: dict[Label, HabiroElement] = {}
        self.acc_hi: dict[Label, HabiroElement] = {}

    def _fold(self, rho_s: dict[Label, HabiroElement],
              acc: dict[Label, HabiroElement], c: Label, fc: LaurentPoly) -> None:
        for s_lbl, g_s in rho_s.items():
            prod = self.aux.multiply(s_lbl, c)        # left = ρ_IR(s), right = c
            for d, c_sc in prod.terms.items():
                if c_sc.is_zero():
                    continue
                contrib = g_s * (fc * c_sc)
                if contrib.is_zero():
                    continue
                acc[d] = (acc[d] + contrib) if d in acc else contrib

    def add(self, c: Label, fc: LaurentPoly) -> None:
        """Fold the new term `fc·L_c` of `RG(a)` into the cutoff acc(s)."""
        self._fold(self.rho_s_lo, self.acc_lo, c, fc)
        if self.rho_s_hi is not None:
            self._fold(self.rho_s_hi, self.acc_hi, c, fc)

    @staticmethod
    def _nonpos(acc: dict[Label, HabiroElement]) -> dict[Label, dict[int, int]]:
        out: dict[Label, dict[int, int]] = {}
        for d, h in acc.items():
            if h.is_zero():
                continue
            np = {e: co for e, co in h.expand(0)._coeffs.items() if e <= 0}
            if np:
                out[d] = np
        return out

    def test(self) -> Label | None:
        """`upper` iff `RG(a)` is complete, else `None`.

        This is the completion certificate for the **windowed** `solve_rg` (the
        q-order `rg_generator` path, used when the grading has no per-charge
        oracle).  Two certificates:

        * **Cutoff stability** (when `rho_s_hi` is set): complete iff the
          non-positive entries identical at both cutoffs are exactly one
          `{0:1}` (`= ρ_UV⁻¹(a)`); the deep-edge artifacts move with the
          cutoff and are filtered.  Robust for any grading.
        * **Gap heuristic** (neither): top entry `{0:1}` isolated by a height
          gap — correct for the `deg = id` / quantum-torus corner.

        (A single-cutoff "drop off-cone ρ-artifacts" filter via
        `grading.cone_gens` was tried but is not robust — the orthant cone is
        too strict on the degree-0 / survivor charges, and a plain height
        threshold can miss a premature residual that dips below it.  The
        cone generators are used for the `_s_rg_component` grading-height
        *windowing* instead.)"""
        lo = self._nonpos(self.acc_lo)
        if self.rho_s_hi is not None:
            hi = self._nonpos(self.acc_hi)
            # Genuine (cutoff-stable) entries: identical non-positive in both
            # cutoffs.  The deep-edge truncation artifacts move with the cutoff,
            # so they drop out; a *premature* `RG(a)` leaves a stable
            # non-positive residual chain (more than one stable entry).  Complete
            # iff the single stable entry is the clean upper `{0:1}`.
            stable = {d: np for d, np in lo.items() if hi.get(d) == np}
            if len(stable) == 1:
                (d, np_d), = stable.items()
                return d if np_d == {0: 1} else None
            return None
        # Single-cutoff gap heuristic (deg=id corner).
        ents = sorted(((self.grading.height_of(d), d, np) for d, np in lo.items()),
                      key=lambda t: t[0])
        if not ents:
            return None
        top_h, top_d, top_np = ents[-1]
        if top_np != {0: 1}:
            return None
        if len(ents) >= 2 and ents[-2][0] > top_h - 2:
            return None
        return top_d



def solve_rg(
    aux: KAlgebra,
    grading: Grading,
    s_rg: dict[Label, HabiroElement],
    apex: Label,
    *,
    s_rg_hi: dict[Label, HabiroElement] | None = None,
    max_height: int | None = None,
    max_labels: int = 100_000,
) -> tuple[Element, Label | None]:
    """Solve `RG(a)·S_RG = L_apex + O(q)` exactly, and derive `ρ_UV⁻¹(a)`.

    Strategy (design session): walk the discovery relation **up** the
    positive cone building `RG(a)`; after each correction, test the
    **mirror constraint** `ρ_IR(S_RG)·RG(a) = L_upper + O(q)` (which the
    *same* `RG(a)` must satisfy, anchored at the top).  When it holds,
    `RG(a)` is complete — stop — and the single upper element gives
    `ρ_UV⁻¹(a) = ρ_IR⁻¹(upper)`.  This is what makes the walk terminate:
    the forward peel alone cannot tell the true top from a continuation
    spuriously produced by the finite charge-support of `S_RG` (it
    over-climbs the cone forever), so the mirror is what bounds it.

    Completion is certified by **cutoff stability** (`_MirrorAcc`): the
    mirror is evaluated at the given `S_RG` *and* a higher-cutoff `S_RG`
    (`s_rg_hi`), and `RG(a)` is complete iff the cutoff-*stable*
    non-positive entries are exactly one `{0:1}` (the deep-edge truncation
    artifacts move with the cutoff and are filtered out).  This replaces
    the old "isolated by a height gap" heuristic, which both closed early
    (truncated `RG`) and failed to close (infinite climb).  If `s_rg_hi`
    is omitted it defaults to `s_rg` (the stability filter then degrades
    to "single clean `{0:1}`" — adequate only when the artifacts already
    sit at distinct labels; callers should pass a genuinely
    higher-cutoff `s_rg_hi`).

    Parameters
    ----------
    aux
        The auxiliary (IR) `KAlgebra` — `multiply`, `identity`, `rho`,
        `rho_inverse`.
    grading
        The `Γ_RG`-grading of `aux`; walks labels in increasing height.
    s_rg
        `S_RG` as `{aux label: exact HabiroElement}` (e.g.
        `rg_generator(cutoff)`); `[S_RG]_0 = 1_B` required.
    apex
        The apex / lower-tropical target label.
    s_rg_hi
        A higher-cutoff `S_RG` (`rg_generator(cutoff + Δ)`) for the
        cutoff-stability completion certificate.  Defaults to `s_rg`.
    max_labels
        Safety cap on processed labels.

    Returns
    -------
    (Element, Label | None)
        `RG(a)` (palindromic `Z[q^±]` coefficients) and the derived label
        `ρ_UV⁻¹(a)` (or `None` if the mirror constraint never closed
        within the cap).
    """
    e_id = aux.identity()

    # Split S_RG: identity part (must be the exact monomial 1) and the
    # height>0 rest (the forward propagation kernel).
    s_hpos: dict[Label, HabiroElement] = {}
    id_coeff = None
    for lbl, co in s_rg.items():
        if co.is_zero():
            continue
        if lbl == e_id:
            id_coeff = co
            continue
        if grading.height_of(lbl) <= 0:
            raise ValueError(
                f"S_RG has non-identity support {lbl!r} at height "
                f"{grading.height_of(lbl)} ≤ 0; the degree-0 part of S_RG "
                f"must be exactly the identity ([S_RG]_0 = 1_B)."
            )
        s_hpos[lbl] = co
    if id_coeff is None or id_coeff.numerator._coeffs != {0: 1} or id_coeff.denom:
        raise ValueError(
            f"S_RG identity coefficient must be the exact monomial 1; "
            f"[S_RG]_0 = 1_B is required for the triangular solve."
        )

    # ρ_IR(S_RG) for the mirror constraint (full generator, identity kept),
    # at both the given cutoff and a higher one (for the cutoff-stability
    # completion certificate).
    def _rho_of(s: dict[Label, HabiroElement]) -> dict[Label, HabiroElement]:
        out: dict[Label, HabiroElement] = {}
        for lbl, co in s.items():
            if co.is_zero():
                continue
            rl = aux.rho(lbl)
            out[rl] = (out[rl] + co) if rl in out else co
        return out

    rho_s_lo = _rho_of(s_rg)
    rho_s_hi = _rho_of(s_rg_hi) if s_rg_hi is not None else None

    h0 = grading.height_of(apex)

    f: dict[Label, LaurentPoly] = {apex: LaurentPoly({0: 1})}
    # Exact residual [RG · (height>0 part of S)]_label as HabiroElement —
    # NO q-truncation, so cancellations are exact.
    resid: dict[Label, HabiroElement] = {}
    processed: set[Label] = set()

    heap: list[tuple[int, int, Label]] = []
    counter = 0
    heapq.heappush(heap, (h0, counter, apex))

    def _push(lbl: Label) -> None:
        nonlocal counter
        h = grading.height_of(lbl)
        if max_height is not None and h > max_height:
            return                                 # window can't reach here
        counter += 1
        heapq.heappush(heap, (h, counter, lbl))

    # Incremental two-cutoff mirror evaluator (cutoff-stable completion):
    # maintains ρ_IR(S_RG)·f at both cutoffs, updated by each newly-added
    # f-term (apex-seeded), so completion-testing is incremental.
    mirror = _MirrorAcc(aux, rho_s_lo, rho_s_hi, grading)
    mirror.add(apex, f[apex])

    def _build(upper: Label | None) -> tuple[Element, Label | None]:
        rg = Element({l: c for l, c in f.items() if not c.is_zero()})
        rho_uv_inv = aux.rho_inverse(upper) if upper is not None else None
        return rg, rho_uv_inv

    # The seeded RG may already be complete (apex-only RG, e.g. a node).
    upper = mirror.test()
    if upper is not None:
        return _build(upper)

    n_done = 0
    while heap:
        h_c, _, c = heapq.heappop(heap)
        if c in processed:
            continue
        processed.add(c)
        n_done += 1
        if n_done > max_labels:
            raise RuntimeError(
                f"solve_rg exceeded max_labels={max_labels}; the mirror "
                f"constraint never closed (check ρ_IR, the grading, or "
                f"that [S_RG]_0 = 1_B with exact Habiro coefficients)."
            )

        # --- solve f[c] (palindromic) -----------------------------------
        if c == apex:
            fc_lp = f[apex]
            added = False
        else:
            r = resid.get(c)
            if r is None or r.is_zero():
                continue                         # residual already O(q): f[c]=0
            nonpos = {e: co for e, co in r.expand(0)._coeffs.items() if e <= 0}
            corr = _peel_nonpositive(nonpos)
            if not corr:
                continue                         # f[c] = 0; nothing to do
            fc_lp = LaurentPoly(corr)
            f[c] = fc_lp
            added = True

        # --- propagate f[c] through the height>0 part of S (exact) ------
        for s_lbl, g_s in s_hpos.items():
            prod = aux.multiply(c, s_lbl)
            for d, c_ts in prod.terms.items():
                if c_ts.is_zero():
                    continue
                contrib = g_s * (fc_lp * c_ts)
                if contrib.is_zero():
                    continue
                lead = _leading_q(contrib)         # skip-positive gate
                if lead is not None and lead > 0:
                    continue
                resid[d] = (resid[d] + contrib) if d in resid else contrib
                if d not in processed:
                    _push(d)

        # --- mirror-constraint termination ------------------------------
        if added:
            mirror.add(c, fc_lp)
            upper = mirror.test()
            if upper is not None:
                return _build(upper)

    return _build(mirror.test())


# ---------------------------------------------------------------------------
# Exact per-charge solve (the oracle path — no window, no mirror)
# ---------------------------------------------------------------------------


def solve_rg_exact(
    aux: KAlgebra,
    grading: Grading,
    s_rg_component,
    apex: Label,
    *,
    max_labels: int = 100_000,
) -> tuple[Element, Label | None]:
    """Solve `RG(a)·S_RG = L_apex + O(q)` with the **per-charge oracle**.

    `s_rg_component(γ) → {aux label: HabiroElement}` is the exact, finite,
    off-cone-vanishing `Γ_RG`-graded component `[S_RG]_γ` (with
    `[S_RG]_0 = 1_B`).  Unlike `solve_rg` — which takes a *fixed window* of
    `S_RG` and needs the mirror / negative-cone completion certificate to
    bound the over-climb the window's truncation causes — this computes every
    residual **exactly** by fetching precisely the `S_RG` pieces that
    contribute to it:

        [RG(a)·S_RG]_d  =  f[d]  +  Σ_{c : h(deg d − deg c) > 0}
                                       f[c] · [S_RG]_{deg d − deg c} |_d ,

    a *finite* exact sum over the current `RG(a) = f`.  Because there is no
    truncation, beyond the true top the residual is **exactly `O(q)`** (exact
    cancellation), the forward peel pushes nothing further, and the cone-walk
    **terminates on its own** — the per-charge oracle makes the whole
    window/mirror/stability apparatus unnecessary.  This is the user's
    principle taken to its conclusion: "you are given a function which computes
    the `γ` part of `S_RG` for every `γ ∈ Γ_RG` — use it."

    `ρ_UV⁻¹(a)` is read off as `ρ_IR⁻¹(top)`, `top` the highest-height `RG(a)`
    label (the leading, coefficient-`1` term — the mirror's single upper
    element).  Triangularity: charges are walked in increasing height, so when
    a charge `γ` is processed every `c` with `h(deg γ − deg c) > 0` (lower
    height) is already solved.  Walk grows only from charges that produced a
    term, so the explored set is the (finite) RG support plus its one-step
    cone frontier.
    """
    g = grading
    if g.cone_gens is None:
        raise ValueError(
            "solve_rg_exact needs grading.cone_gens (the positive-cone "
            "generators that drive the per-charge cone walk)."
        )
    zero = g.zero_charge()

    f: dict[Label, LaurentPoly] = {apex: LaurentPoly({0: 1})}
    apex_charge = g.charge(apex)

    comp_cache: dict[Charge, dict[Label, HabiroElement]] = {}

    def comp(gamma: Charge) -> dict[Label, HabiroElement]:
        c = comp_cache.get(gamma)
        if c is None:
            c = s_rg_component(gamma)
            comp_cache[gamma] = c
        return c

    heap: list[tuple[int, int, Charge]] = []
    counter = 0
    seen_charge = {apex_charge}

    def push_charge(gamma: Charge) -> None:
        nonlocal counter
        if gamma in seen_charge:
            return
        seen_charge.add(gamma)
        counter += 1
        heapq.heappush(heap, (g.h(gamma), counter, gamma))

    for gen in g.cone_gens:
        push_charge(g.charge_sum(apex_charge, tuple(gen)))

    n_done = 0
    while heap:
        _, _, gamma = heapq.heappop(heap)
        n_done += 1
        if n_done > max_labels:
            raise RuntimeError(
                f"solve_rg_exact exceeded max_labels={max_labels}; the "
                f"cone-walk did not terminate (check that `_s_rg_component` is "
                f"exact / finite / vanishes off the cone, and `[S_RG]_0 = 1_B`)."
            )

        # Exact residual at charge `gamma`, summed over the current RG terms.
        R: dict[Label, HabiroElement] = {}
        for c_lbl, fc in f.items():
            gs = tuple(x - y for x, y in zip(gamma, g.charge(c_lbl)))
            if g.h(gs) <= 0 or not g.in_cone(gs):
                continue                          # identity self-term / off-cone
            for s_lbl, g_s in comp(gs).items():
                if g_s.is_zero():
                    continue
                prod = aux.multiply(c_lbl, s_lbl)
                for d, c_ts in prod.terms.items():
                    if c_ts.is_zero():
                        continue
                    contrib = g_s * (fc * c_ts)
                    if contrib.is_zero():
                        continue
                    R[d] = (R[d] + contrib) if d in R else contrib

        produced = False
        for d, r in R.items():
            if r.is_zero():
                continue
            nonpos = {e: co for e, co in r.expand(0)._coeffs.items() if e <= 0}
            corr = _peel_nonpositive(nonpos)
            if not corr:
                continue
            f[d] = LaurentPoly(corr)              # d's charge == gamma (set once)
            produced = True

        if produced:
            for gen in g.cone_gens:
                push_charge(g.charge_sum(gamma, tuple(gen)))

    rg = Element({l: c for l, c in f.items() if not c.is_zero()})
    upper = _mirror_upper_exact(aux, g, comp, f)
    if upper is None:
        upper = max(f, key=lambda l: g.height_of(l))   # fallback: RG top
    return rg, aux.rho_inverse(upper)


def _mirror_upper_exact(aux, g, comp, f, max_labels: int = 100_000) -> "Label | None":
    """Discover the mirror's upper label `ρ_UV⁻¹(a) = ρ_IR⁻¹(upper)` from the
    **negative-cone (mirror) constraint** `ρ_IR(S_RG)·RG(a) = L_upper + O(q)`,
    computed *exactly* via the per-charge oracle.

    `ρ_IR` is the IR ρ-twisted automorphism — a q-linear *label* permutation
    fixing the canonical basis (it permutes labels, leaving coefficients), so
    `ρ_IR(S_RG) = Σ_p Σ_{s∈[S_RG]_p} co_s · L_{ρ(s)}` and
    `M = ρ_IR(S_RG)·RG(a) = L_upper + O(q)`, with `upper` the unique label
    whose non-positive-`q` part is `{0:1}` (all other labels `O(q)`).

    Computed exactly the way the forward solve is — walk the **output**
    charges `d`, not the (infinite) spectrum charges `p`: each mirror
    coefficient `M[d]` is a *finite* exact sum over the `RG(a) = f` terms `c`,
    with `p = charge(c) − d` the (unique) spectrum charge that lands there.
    `ρ_IR(S_RG)` lives in the negative cone, so `M` descends from the
    `f`-charges; the walk starts at the `f`-charges (`p=0` ⇒ `M ⊇ f`) and
    steps **down** the cone, stopping when a whole layer is `O(q)` (the tail is
    `O(q)`).  This is what makes the RG solver discover the *genuine* flow-`ρ`
    (not merely the RG top, which equals `upper` only when the survivor is the
    whole IR — the quantum-torus corner; with a central flavour they differ).
    """
    if not f:
        return None
    M: dict[Label, HabiroElement] = {}
    # The upper (a ρ-image of an `RG(a)` support label) lies within the
    # spectrum-charge span of `f`; bound the descent there so the walk
    # terminates (below it `M` is `O(q)` but the ρ_IR(S_RG) negative-cone tail
    # would otherwise keep the per-charge contributions non-`O(q)`).
    apex_h = min(g.height_of(c) for c in f)

    def compute_d(d_charge: Charge) -> bool:
        """Fill `M[d]` for every label `d` at charge `d_charge`; return whether
        any such label has a non-positive-`q` part (⇒ keep descending)."""
        local: dict[Label, HabiroElement] = {}
        for c_lbl, fc in f.items():
            p = tuple(x - y for x, y in zip(g.charge(c_lbl), d_charge))
            if g.h(p) < 0 or not g.in_cone(p):
                continue
            for s_lbl, co in comp(p).items():
                if co.is_zero():
                    continue
                prod = aux.multiply(aux.rho(s_lbl), c_lbl)   # ρ_IR(L_s)·L_c
                for d, c_ts in prod.terms.items():
                    if c_ts.is_zero():
                        continue
                    contrib = co * (fc * c_ts)
                    if contrib.is_zero():
                        continue
                    local[d] = (local[d] + contrib) if d in local else contrib
        any_nonpos = False
        for d, h in local.items():
            M[d] = h
            if h.is_zero():
                continue
            if any(e <= 0 for e in h.expand(0)._coeffs):
                any_nonpos = True
        return any_nonpos

    heap: list[tuple[int, int, Charge]] = []
    counter = 0
    seen: set[Charge] = set()

    def push(dc: Charge) -> None:
        nonlocal counter
        if dc in seen or g.h(dc) < apex_h:           # bound descent to f-span
            return
        seen.add(dc)
        counter += 1
        heapq.heappush(heap, (-g.h(dc), counter, dc))   # descend: highest first

    for c_lbl in f:                                       # seed at the f-charges
        push(g.charge(c_lbl))

    n_done = 0
    while heap:
        _, _, dc = heapq.heappop(heap)
        n_done += 1
        if n_done > max_labels:
            return None
        if compute_d(dc):
            for gen in g.cone_gens:
                push(tuple(x - y for x, y in zip(dc, tuple(gen))))   # step down

    upper = None
    for d, h in M.items():
        if h.is_zero():
            continue
        nonpos = {e: c for e, c in h.expand(0)._coeffs.items() if e <= 0}
        if nonpos == {0: 1}:
            if upper is not None:
                return None                          # ambiguous → caller fallback
            upper = d
    return upper



def solve_trg_exact(
    aux: KAlgebra,
    grading: Grading,
    s_rg_component,
    apex: Label,
    *,
    max_labels: int = 100_000,
) -> tuple[Element, Label | None]:
    """`tRG(a)` and forward `ρ_UV(a)` via the exact per-charge oracle — the
    `solve_rg_exact` of the **left** discovery relation `S_RG·tRG = L_apex +
    O(q)`, i.e. `solve_rg_exact` in the opposite algebra (cf. `solve_trg`)."""
    return solve_rg_exact(_OppositeKAlgebra(aux), grading, s_rg_component, apex,
                          max_labels=max_labels)


# ---------------------------------------------------------------------------
# The alternative RG map  tRG  and forward ρ_UV
# ---------------------------------------------------------------------------


class _OppositeKAlgebra(KAlgebra):
    """The opposite algebra of `aux`: same objects, `multiply` swapped
    (`x ·_op y = y ·_aux x`).  Everything else delegates to `aux`.

    Used to solve the **left** discovery relation `S_RG·tRG(a) = a + O(q)`
    by reusing the (right-multiplying) `solve_rg`: in the opposite algebra
    that relation reads `tRG(a) ·_op S_RG = a + O(q)`, the ordinary
    discovery relation.
    """

    def __init__(self, aux: KAlgebra):
        self._aux = aux

    def coefficient_ring(self):
        return self._aux.coefficient_ring()

    def identity(self):
        return self._aux.identity()

    def multiply(self, x, y):
        return self._aux.multiply(y, x)

    def rho(self, a):
        return self._aux.rho(a)

    def rho_inverse(self, a):
        return self._aux.rho_inverse(a)

    def trace(self, a, K: int = 20):
        return self._aux.trace(a, K)

    def r_label_decompose(self, label):
        # Same label space as the auxiliary — delegate the lift coordinate
        # (raises iff the auxiliary has not been migrated).
        return self._aux.r_label_decompose(label)

    def r_label_compose(self, section, r_basis_label):
        return self._aux.r_label_compose(section, r_basis_label)



def solve_trg(
    aux: KAlgebra,
    grading: Grading,
    s_rg: dict[Label, HabiroElement],
    apex: Label,
    *,
    s_rg_hi: dict[Label, HabiroElement] | None = None,
    max_height: int | None = None,
    max_labels: int = 100_000,
) -> tuple[Element, Label | None]:
    """The **alternative RG map** `tRG(a)` and forward `ρ_UV(a)`.

    `tRG(a)` is defined by `RG(a)·S_RG = S_RG·tRG(a)`, i.e.
    `tRG(a) = S_RG⁻¹·RG(a)·S_RG = ρ_IR⁻¹(RG(ρ_UV(a)))` (the conjugate of
    `RG(a)` by `S_RG`).  It satisfies the **left** discovery relation
    `S_RG·tRG(a) = L_apex + O(q)`, which is the ordinary discovery
    relation in the **opposite algebra** — so it is solved by `solve_rg`
    over `_OppositeKAlgebra(aux)`, whose mirror constraint then yields
    forward `ρ_UV(a)` directly.

    Returns `(tRG(a), ρ_UV(a))`.  The intertwining
    `RG ∘ ρ_UV = ρ_IR ∘ tRG` ties this to the forward solve.
    """
    return solve_rg(_OppositeKAlgebra(aux), grading, s_rg, apex,
                    s_rg_hi=s_rg_hi,
                    max_height=max_height, max_labels=max_labels)
