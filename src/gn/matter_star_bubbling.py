"""`matter_star_bubbling` — the constraint calculus for `L_{m,e}` of `(G, N)`.

The matter-side companion of `star_bubbling`.  Where the pure solver pins a
canonical from **(★) + bar + `O(𝖖)`**, a theory with `T^*N` matter has a
**fourth, independent** condition — per-cell numerator divisibility — and this
module formulates it, makes it executable, and uses it to constrain the build.

The four conditions, and what each one does
-------------------------------------------

| condition | lives on | role for `(G, N)` |
|---|---|---|
| bar (W1) | palindromy of each residual | ansatz-level (built in) |
| (★) pole cancellation | gauge walls `v^α = 𝖖^{−k}` | **determines — at least as much as in `(G,0)`, and strictly more** |
| `O(𝖖)` bubbling | subleading cells `a_m(v)` | determines |
| **μ-divisibility** | matter poles `μ_i v^w = −𝖖^{2s−1}` | **determines + guards** |

**(★) is NOT empty here — correcting an earlier claim in this file.**
the design notes measured `T_a(Z_a)|wall == T_p(Z_p)|wall` on
affine-Weyl partner pairs and concluded `Z` factors out of the wall equation.
That is true **pair-by-pair exactly when `Δ_N(a) = Δ_N(p)`** — the sampled
family (U(2), `m = (0,−2),(0,−3),(−1,−3),(−1,−4)`) consists of such pairs.  It
is *not* a property of the wall equations in general.  Measured
(a probe in the source repository), counting partner pairs with unequal
`Δ_N`:

    U(2)  m=(0,−2)   4 equal / 2 DIFFERENT      SU(2) m=(−2,)   5 / 20
    U(2)  m=(0,−3)   8 equal / 6 DIFFERENT      SU(2) m=(−3,)   8 / 48
    SU(3) m=(−1,0)  24 equal / 24 DIFFERENT

Even at U(2), where all *atoms* carry equal `Δ_N`, the partners reach **outside
the support** (`(0,−2) ↔ (−3,1)`, `Δ_N` 2 vs 3), where the partner residual is
zero and the wall condition falls on `d_a` alone.

Consequences for the constructor.  Writing `f_a = Z_a·Q_a`, the wall residue of
`d_a + d_p` is `T_a(Z_a)·res(d^Q_a) + T_p(Z_p)·res(d^Q_p)`.  Where the two `Z`'s
agree it collapses to (★) on the `Q`'s — the *same* system the pure solver
uses.  Where they differ, the two sides carry different μ-degrees, so the
condition **couples the μ-levels** — work (★) does not do in `(G,0)` at all.
Either way it is at least as strong as in the pure theory, and it is the
condition the bubbled-cell tail `Q[j≥1]` must be solved against.

The divisibility condition
--------------------------
Per cell `m'`, the no-matter-denominator property of `S_RG` (derived in
the design notes from a property weaker than the discovery relation)
says `RG(a)_{m'}·U_{m'}·S_RG` carries no matter denominators.  Since
`S_RG(𝖖^{2m'}v)/S_RG(v) = 1/Z_{m'}(…)`, that is a **divisibility of the cell's
μ-polynomial**:

    F_{m'}(μ) := Σ_k μ^k f_{m',k}(v)   is divisible in μ by
    Z_{m'}(μ)  =  ∏_{w: c<0} ∏_{s=0}^{|c|−1} (1 + μ 𝖖^{2s+c+1} v^w),
    c := ⟨m', w⟩                                        (bare `v`; measured)

**Measured (2026-07-27), 22/22 cells across SU(2) `m=(−2,),(−3,)`, SU(3)
`m=(−1,0)`, U(2) `m=(0,−2)` — extremal AND bubbled alike:**

* divisibility holds on **every** cell;
* the quotient obeys the degree law

      deg_μ Q_{m'}  =  Δ_N(m) − Δ_N(m'),     Δ_N(n) := Σ_w max(0, −⟨n,w⟩)

  and the whole element's top μ-level is `Δ_N(m)`;
* `Q_{m'}[μ⁰]` is the **pure** residual at that cell, at every cell;
* on extremal cells `Δ_N` is maximal, so `deg_μ Q = 0` and `Q` *is* the pure
  residual — which recovers the earlier leading-orbit product law
  `F = pure · Z` as the special case of this general statement.

So `F = Z·Q` is the right ansatz: it satisfies divisibility by construction,
its `μ⁰` coefficient is known, and the only free parameters are `Q[μ^j]`
(`j ≥ 1`) on cells where `Δ_N` drops below its leading value — i.e. exactly the
bubbled cells, and exactly as many parameters as the degree law allows.

**D2 is a HEURISTIC, not an axiom**.  It is used
here to *decide what to try*, never as the ground of correctness: where the
degree law says a μ-level carries no free parameter, `solve_level` is skipped
and `f = offset` proposed — but the proposal is then put through
`verify_forced_level`, i.e. **(★) + W1**, the same axioms that license the solve
and which know nothing about D2.  A proposal that fails falls through to the
ordinary solve.  This matters because the strict divisibility guard cannot play
that role in the forced levels: it recomputes the quotient degree from what was
built and compares it to the bound used to build it, which is circular there.

Scope: the conditions and the guard are datum-general; see
the design notes for the ladder this belongs to.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from root_datum import RootDatum
from weyl_torus_ring import TorusLaurent, TorusRational


__all__ = [
    "verify_forced_level",
    "delta_N",
    "matter_monomials",
    "Z_levels",
    "divide_by_linear",
    "divide_by_Z",
    "check_divisibility",
    "DivisibilityFailure",
    "MatterCellReport",
]


class DivisibilityFailure(RuntimeError):
    """A cell's μ-polynomial is not divisible by its matter dressing `Z`."""


# ---------------------------------------------------------------------------
# The matter data at a cell
# ---------------------------------------------------------------------------


def _weights(datum: RootDatum, matter):
    from g_matter_over_pure import matter_weights, _normalize_matter
    out = []
    for lam in _normalize_matter(datum, matter):
        out.append(matter_weights(datum, lam))
    return out


def delta_N(datum: RootDatum, matter, n) -> int:
    """`Δ_N(n) = Σ_i Σ_{w ∈ wt(N_i)} max(0, −⟨n, w⟩)` — the matter-pole count at
    cell `n`, and (measured) the top μ-level of `L_{n,e}` when `n` is the
    label's magnetic charge.

    This is the invariant that decides where the leading-orbit product law
    `F = pure · Z` holds: it holds exactly where `Δ_N` attains its
    leading-orbit value, and the quotient degree elsewhere is the shortfall."""
    tot = 0
    for wts in _weights(datum, matter):
        for w in wts:
            c = matter_pairing(datum, n, w)
            if c < 0:
                tot += -c
    return tot


def matter_pairing(datum, n, w) -> int:
    """`⟨n, w⟩` as a Python **int** — the matter-cell pairing, used throughout as a
    count of zero modes (`range(-c)`) and as a `𝖖`-exponent.

    A cocharacter of a non-simply-connected global form has `Fraction`
    coordinates, so this comes back a `Fraction` even when integral, and `range()`
    rejects that on type alone.  For a **legal** matter representation of the form
    every weight lies in the form's electric lattice, so `⟨n, w⟩ ∈ Z` and the
    coercion is always sound.

    Honest-fails otherwise: a non-integral value means the matter carries centre
    charge against a fractional coweight, i.e. it is not a representation of this
    global form — which `GNAbeKAlgebra` rejects up front, so reaching this error
    means the caller bypassed that check."""
    c = sum(x * y for x, y in zip(n, w))
    ic = int(c)
    if ic != c:
        raise NotImplementedError(
            f"{datum.name}: ⟨n, w⟩ = {c} is not an integer at cell {tuple(n)}, "
            f"matter weight {tuple(w)} — the matter is not a representation of "
            f"this global form (a centre-charged weight against a fractional "
            f"coweight).")
    return ic


def matter_monomials(datum: RootDatum, matter, n) -> list:
    """The factors of `Z_n(μ) = ∏ (1 + μ·x)` at cell `n`, as
    `[(slot, weight, 𝖖-exponent)]` with exponent `2s + ⟨n,w⟩ + 1` (bare `v` —
    the shifted candidate `2s + 2⟨n,w⟩ + 1` is measured to fail)."""
    out = []
    for i, wts in enumerate(_weights(datum, matter)):
        for w in wts:
            c = matter_pairing(datum, n, w)
            if c >= 0:
                continue
            for s in range(-c):
                out.append((i, tuple(w), 2 * s + c + 1))
    return out


def Z_levels(datum: RootDatum, matter, n, kmax: int | None = None) -> dict:
    """`{k: TorusLaurent}` — the `μ^k` coefficients of `Z_n(μ)`, i.e. the
    elementary symmetric polynomials in the cell's matter monomials."""
    monos = matter_monomials(datum, matter, n)
    top = len(monos) if kmax is None else min(kmax, len(monos))
    levels = {0: TorusLaurent.one(datum)}
    for (_i, w, E) in monos:
        x = TorusLaurent.monomial(datum, tuple(w), LaurentPoly({E: 1}))
        nxt = {}
        for k, v in levels.items():
            nxt[k] = nxt.get(k, TorusLaurent.zero(datum)) + v
            if k + 1 <= top:
                nxt[k + 1] = nxt.get(k + 1, TorusLaurent.zero(datum)) + v * x
        levels = nxt
    return {k: v for k, v in levels.items() if k <= top}


# ---------------------------------------------------------------------------
# Division in μ
# ---------------------------------------------------------------------------


def divide_by_linear(datum, coeffs: dict, x: TorusRational):
    """Divide `F(μ) = Σ_k coeffs[k]·μ^k` by `(1 + μ·x)`.

    Division proceeds from the **bottom**: `(1 + μx)·Σ_j q_j μ^j` has `μ⁰`
    coefficient `q_0` and `μ^j` coefficient `q_j + x·q_{j−1}`, so
    `q_0 = c_0`, `q_j = c_j − x·q_{j−1}`, and the remainder is the leftover top
    coefficient.  (Dividing from the top would need to invert `x`, which is a
    monomial and would silently succeed on non-divisible input.)

    Returns `(quotient: dict, remainder: TorusRational)`."""
    if not coeffs:
        return {}, TorusRational.zero(datum)
    n = max(coeffs)
    c = [coeffs.get(k, TorusRational.zero(datum)) for k in range(n + 1)]
    q, prev = [], None
    for j in range(n):
        qj = c[j] if prev is None else (c[j] + (prev * x) * (-1)).simplify()
        q.append(qj)
        prev = qj
    rem = ((c[n] + (prev * x) * (-1)).simplify() if prev is not None
           else c[n])
    return {j: v for j, v in enumerate(q)}, rem


def divide_by_Z(datum, matter, n, coeffs: dict):
    """Divide a cell's μ-polynomial by `Z_n(μ)`, factor by factor.

    Returns `(quotient: dict, ok: bool)`.  `ok` is the **divisibility
    condition** at this cell — the executable form of the no-matter-denominator
    property."""
    work = dict(coeffs)
    for (_i, w, E) in matter_monomials(datum, matter, n):
        x = TorusRational.from_laurent(
            TorusLaurent(datum, {tuple(w): LaurentPoly({E: 1})}))
        work, rem = divide_by_linear(datum, work, x)
        if not rem.is_zero():
            return work, False
    return {j: v for j, v in work.items() if not v.is_zero()}, True


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


class MatterCellReport:
    """Per-cell outcome of the divisibility guard."""

    __slots__ = ("cell", "extremal", "delta", "levels", "divisible",
                 "quotient", "quotient_degree", "quotient_is_pure")

    def __init__(self, cell, extremal, delta, levels, divisible, quotient,
                 quotient_degree, quotient_is_pure):
        self.cell = cell
        self.extremal = extremal
        self.delta = delta
        self.levels = levels
        self.divisible = divisible
        self.quotient = quotient
        self.quotient_degree = quotient_degree
        self.quotient_is_pure = quotient_is_pure

    def __repr__(self):
        return (f"MatterCell({self.cell}, "
                f"{'extremal' if self.extremal else 'bubbled'}, "
                f"Δ_N={self.delta}, divisible={self.divisible}, "
                f"deg Q={self.quotient_degree})")


def check_divisibility(datum, matter, m, cells: dict, pure=None,
                       strict: bool = False):
    """The matter guard on a candidate `L_{m,e}` presented cell-wise.

    `cells` is `{atom: {μ-level: TorusRational}}` — e.g. the transpose of
    `GMatterOverPure.rg_chart`.  `pure`, if given, is `{atom: TorusRational}`
    for the corresponding pure canonical, enabling the `Q[μ⁰] == pure` check.

    Returns `{atom: MatterCellReport}`.  With `strict=True`, raises
    `DivisibilityFailure` on the first cell that fails divisibility or the
    degree law.

    The degree law checked here is the measured
    `deg_μ Q_{m'} = Δ_N(m) − Δ_N(m')`."""
    lead = delta_N(datum, matter, tuple(m))
    out = {}
    orbit = {tuple(datum.act_cochar(w, tuple(m)))
             for w in datum.weyl_elements()}
    for atom, levels in cells.items():
        atom = tuple(atom)
        dn = delta_N(datum, matter, atom)
        quot, ok = divide_by_Z(datum, matter, atom, levels)
        deg = max(quot) if quot else 0
        is_pure = None
        if pure is not None and atom in pure:
            is_pure = (0 in quot
                       and quot[0].simplify() == pure[atom].simplify())
        rep = MatterCellReport(atom, atom in orbit, dn, sorted(levels), ok,
                               quot, deg, is_pure)
        out[atom] = rep
        if strict:
            if not ok:
                raise DivisibilityFailure(
                    f"cell {atom}: μ-polynomial not divisible by Z "
                    f"(Δ_N={dn})")
            if deg != lead - dn:
                raise DivisibilityFailure(
                    f"cell {atom}: quotient degree {deg} != "
                    f"Δ_N(m)−Δ_N(m') = {lead - dn}")
    return out


def solve_level(datum, matter, m, e, n, pad: int = 1, lines=None,
                verbose: bool = False, oq_offset=None, max_pad_retry: int = 2,
                audit=None, support=None, stats=None, admits_e=None):
    """Solve the **μ-level-`n` slice** of `L_{m,e}` for `(G, N)` from
    (★) + bar + `O(𝖖)`, reusing the pure solver.

    The slice is a well-posed instance of the pure system because **(★) holds
    separately at every μ-level** — measured on the flow's own `RG(a)`, 10/10
    levels across SU(2) `m=(−1,),(−2,)`, SU(3) `(−1,0)`, U(2) `(0,−2)`.  Two
    things change relative to the pure problem:

    * the **seed** on extremal cells is `pure_a · Z_a[n]` (the leading-orbit
      product law) instead of the bare leading orbit.  This is asserted, not
      solved for, and that is **correct rather than merely convenient** — the author's
      ruling 2026-07-30: *"the seed written in terms of `U_m` which include matter
      contributions to the `R` cocycle is matter-independent."*  The atoms already
      carry the matter factor `W = T_{−m'}(Z_m)T_m(Z_{m'})/Z_{m+m'}`, so in the
      `U_m` basis the leading orbit has nothing matter-dependent left in it: its
      un-dressed residual IS the pure canonical's, hence `Q[k ≠ 0]|orbit = 0` and
      the extremal cells carry no unknowns for a solve to find.  (The *bubbled*
      cells' degree budget — D2 proper — is a different claim and stays a
      heuristic, guarded by (★)+W1.)  addendum;
    * the **box anchors** are shifted by the weights occurring in `Z[n]`, i.e.
      by sums of `n` matter weights.  These are not roots, so the pure `pad`
      dilation (which runs along root directions) cannot reach them;
    * the `O(𝖖)` condition needs `oq_offset` (supplied by
      `solve_canonical_matter`).  **The threshold is not level-dependent — the
      object it is stated on is.**  Measured: the uniform statement is
      `val_𝖖(Q_interior[j]) ≥ 1` on the QUOTIENT, which holds at every cell and
      every level tested; the naive `val_𝖖(f_interior[n]) ≥ 1` is FALSE at
      SU(2) `m=(3,)` level 1 and U(2) `m=(0,−3)` levels 1–2 (true valuation 0
      there), and imposing it forces a unique-but-wrong answer.  Since
      `Z[0] = 1`, `f[n] = Q[n] + Σ_{i≥1} Z[i]·Q[n−i]`, so the correction is a
      known constant once the lower levels are solved.

    Returns `(status, Ff, interior)` exactly as `star_bubbling.joint_solve`.
    At `n = 0` the seed is the plain leading orbit and this reduces to the pure
    solve.

    `support`, `stats` and `admits_e` (default `None` = unchanged) are handed to
    `star_bubbling.joint_solve` as they are — the support condition measured
    rather than assumed (see `matter_multislot.solve_canonical_matter_vec`).
    With `support`, the box anchors and the 𝖖-window also cover the dressing
    `Z[n]` on the cells outside the hull: a cutoff that could not hold a filling
    of those cells would report the vanishing it is there to test."""
    import star_bubbling as SB
    from pure_g_abe_kalgebra import PureGAbeKAlgebra

    m, e = tuple(m), tuple(e)
    # The 4d gauge group data must be threaded in: the `μ⁰` slice is a PURE
    # canonical, so this builds its own PureGAbeKAlgebra, and without `lines`
    # that silently defaults to the SIMPLY CONNECTED form and refuses the
    # fractional coweights a non-simply-connected form is defined by.
    P = PureGAbeKAlgebra(datum, lines=lines)
    pure = P.chart((m, e))
    pres = pure.residuals()
    orbit = {tuple(datum.act_cochar(w, m)) for w in datum.weyl_elements()}

    # `n` may be an int (total μ-degree, M = 1) or a multi-index `k⃗`
    # (one component per hypermultiplet slot, M >= 1 — see `matter_multislot`).
    if isinstance(n, tuple):
        from matter_multislot import Z_levels_vec

        def _Z(cell):
            return Z_levels_vec(datum, matter, cell, kmax=n).get(n)
    else:
        def _Z(cell):
            return Z_levels(datum, matter, cell, kmax=n).get(n)

    hull = [tuple(p) for p in SB.tropical_support(datum, m)]
    cells = hull if support is None else [tuple(p) for p in support]
    beyond = [] if support is None else [c for c in cells if c not in hull]
    seed_f, z_weights, z_qmax = {}, set(), 0
    for a in list(pres) + beyond:
        Zn = _Z(a)
        if a in orbit:
            if Zn is None:
                seed_f[a] = TorusRational.zero(datum)
            else:
                seed_f[a] = (pres[a] * TorusRational.from_laurent(Zn)
                             ).simplify()
        if Zn is not None:
            z_weights.update(Zn.terms)
            # The 𝖖-degrees of `Z[k⃗]`, for the 𝖖-window — the exact analogue of
            # collecting its WEIGHTS for the box anchors just above.
            for _wt, lp in Zn.terms.items():
                for dq, c in lp._coeffs.items():
                    if c:
                        z_qmax = max(z_qmax, abs(dq))

    # No bubbling cells (minuscule / central `m`) ⇒ nothing to solve: the seed
    # IS the answer.  `joint_solve` has no unknowns to build an ansatz from and
    # raises there, so short-circuit rather than let it fail.
    if not [p for p in cells if p not in orbit]:
        return "unique", {}, []
    extra = {k: v for k, v in (("support", support), ("stats", stats),
                               ("admits_e", admits_e)) if v is not None}

    anchors = set()
    for base in SB.wilson_weights(datum, e):
        for zw in (z_weights or {(0,) * datum.dim}):
            anchors.add(tuple(base[i] + zw[i] for i in range(datum.dim)))

    # The box is a CUTOFF, never a selector (`star_bubbling`: enlarging is the
    # safe direction, shrinking can silently select), and an `inconsistent`
    # verdict is its documented too-narrow signature — the true answer lies
    # outside the window, so no combination of unknowns can satisfy the seed
    # constants.  Measured at Spin(5) + 2x5, μ-sector (2,2): `inconsistent` at
    # pad=1, solves at pad=2.  Deeper μ-sectors anchor on sums of |k⃗| matter
    # weights, which under-cover the window, so widen on demand rather than
    # honest-failing on what is only a cutoff artifact.
    # WIDEN THE 𝖖-WINDOW BY THE DRESSING'S OWN 𝖖-DEGREES.  `joint_solve`'s
    # default window is `q_extra = 2` above the pure `S(n) − 1`, then widened by
    # the seed constants' degrees.  A matter slice needs more, for exactly the
    # reason its box needs matter-weight anchors: at level `k⃗` the extremal seed
    # is `pure·Z[k⃗]`, so every numerator on the slice is the pure one shifted by
    # `Z[k⃗]`'s content — in the `v`-direction by its WEIGHTS (the anchors, above)
    # and in the 𝖖-direction by its 𝖖-DEGREES (here).  Only the first half was
    # being applied, so the window stayed at its pure value while the required
    # degrees grew with the level.
    #
    # Measured at SU(2)+Adj `m=(3,)`, `e=0`, level 1: the true answer (from
    # `GMatterOverPure`) needs 𝖖-degree 8 at the `(−2,)` cell where `T_of = 7`, so
    # it was not in the ansatz at ANY `pad` — `pad` does not touch the window at
    # all.  That is the sector the full build actually failed on, and it reported
    # the box's too-narrow signature while advising a wider box, which could not
    # have helped.  Enlarging the window is monotone in the same sense as
    # enlarging the box, so it cannot change an answer already pinned.
    q_extra = 2 + z_qmax

    # `audit` short-circuits: report why a KNOWN answer is not found, at this
    # exact seed / anchor / offset construction, rather than re-solving wider.
    if audit is not None:
        return SB.joint_solve(
            datum, m, e, pad=pad, q_extra=q_extra, verbose=verbose,
            seed_f=seed_f, anchors=sorted(anchors), oq_offset=oq_offset,
            audit=audit, **extra)

    for widen in range(0, max_pad_retry + 1):
        status, Ff, _support, interior, _K_f = SB.joint_solve(
            datum, m, e, pad=pad + widen, q_extra=q_extra, verbose=verbose,
            seed_f=seed_f, anchors=sorted(anchors), oq_offset=oq_offset,
            **extra)
        if status != "inconsistent":
            if widen and verbose:
                print(f"  [widened box to pad={pad + widen}]", flush=True)
            return status, Ff, interior
    return status, Ff, interior


def verify_forced_level(datum, lvl, verbose: bool = False):
    """**The degree law proposes; the axioms dispose.**

    D2 is a *measured* law, not an axiom *"I feel D2
    should not be needed a priori, but perhaps it is a useful heuristic."*  So
    where the degree law says a μ-sector carries no free parameter and the
    answer is `f = offset` outright, that answer is **checked against the
    axioms** rather than trusted:

      * **(★)** — affine-Weyl residue cancellation on every wall
        (`star_bubbling.criterion`), i.e. the level's difference operator is
        regular on `Λ = R(G)`.  This is the same guard that licenses the solve;
      * **W1** — bar-invariance of the level.  Bar acts on `𝖖` only and the
        μ-grading is untouched by it, so each level is separately bar-invariant.

    The `O(𝖖)` condition is automatic here: the forced sectors have `Q = 0`, so
    `val_𝖖(Q_interior) = ∞ ≥ 1`.

    Returns `(ok, reason)`.  A caller that gets `ok=False` must fall back to the
    solve rather than emit the proposal — which is what makes D2 a heuristic in
    the code and not an assumption."""
    from wrq_torus import WRQTorus
    import star_bubbling as SB

    x = WRQTorus(datum, {tuple(c): v for c, v in lvl.items()
                         if not v.is_zero()})
    if x.is_zero():
        return True, "zero level"
    if x.bar() != x:
        return False, "W1 fails: level not bar-invariant"
    ok, rep = SB.criterion(x)
    if not ok:
        return False, f"(★) fails: {str(rep)[:120]}"
    return True, "(★)+W1 verified"


def solve_canonical_matter(datum, matter, m, e, pad: int = 1, lines=None,
                           verbose: bool = False):
    """`L_{m,e}` for `(G, N)` as `{μ-level: {cell: TorusRational}}`, built from
    the four conditions — bar + (★) + `O(𝖖)` + μ-divisibility.

    Levels are solved **in order** `0 … Δ_N(m)` ((★) holds per level), each with
    the `O(𝖖)` condition stated on the **quotient** `Q = f/Z` rather than on `f`.
    Since `Z[0] = 1`,

        f[n]  =  Q[n]  +  Σ_{i≥1} Z[i]·Q[n−i],

    so the correction is a known constant once the lower levels are in hand, and
    it enters `joint_solve` as `oq_offset`.  **The threshold itself is uniform**
    — `val_𝖖(Q_interior) ≥ 1` at every cell and level — which is what makes this
    a change of *object*, not a level-dependent threshold.

    The assembled element is then put through the **divisibility guard** in
    strict mode; a level that is not `unique`, or a guard failure, raises rather
    than returning a candidate."""
    m, e = tuple(m), tuple(e)
    from pure_g_abe_kalgebra import PureGAbeKAlgebra

    if not datum.is_dominant_cochar(m):
        raise ValueError(
            f"{datum.name}: m={m} is not cochar-dominant; labels on this tier "
            f"are (m, e) with m cochar-dominant (dominant rep: "
            f"{datum.dominant_cochar_rep(m)})")

    # The 4d gauge group data must be threaded in: the `μ⁰` slice is a PURE
    # canonical, so this builds its own PureGAbeKAlgebra, and without `lines`
    # that silently defaults to the SIMPLY CONNECTED form and refuses the
    # fractional coweights a non-simply-connected form is defined by.
    P = PureGAbeKAlgebra(datum, lines=lines)
    pure = P.chart((m, e))
    pres = pure.residuals()
    orbit = {tuple(datum.act_cochar(w, m)) for w in datum.weyl_elements()}
    top = delta_N(datum, matter, m)
    # D2, the degree law, as a per-cell BUDGET: `deg_μ Q_{m'} = Δ_N(m) −
    # Δ_N(m')`.  Where it is 0 the cell carries no free parameter at that level
    # and `f` is already known — see the short-circuit below.
    budget = {c: top - delta_N(datum, matter, c) for c in pres}

    Q: dict = {}
    out: dict = {}
    for n in range(top + 1):
        offset: dict = {}
        if n:
            for cell in pres:
                if cell in orbit:
                    continue
                Zc = Z_levels(datum, matter, cell, kmax=n)
                acc = TorusRational.zero(datum)
                for i in range(1, n + 1):
                    Zi, Qj = Zc.get(i), Q.get(cell, {}).get(n - i)
                    if Zi is None or Qj is None or Qj.is_zero():
                        continue
                    acc = acc + Qj * TorusRational.from_laurent(Zi)
                offset[cell] = acc.simplify()

        # --- apply the degree law rather than re-deriving it by solve -------
        # If no bubbled cell can carry a free `Q[n]` (D2: `n > Δ_N(m) −
        # Δ_N(m')` everywhere), then `Q[n] = 0` on all of them and
        # `f[n] = offset[n]` outright.  Handing such a level to `joint_solve`
        # anyway asks it to fit unknowns that must not exist, and the
        # over-determined system reports `inconsistent` — measured at U(2)+N_f=2
        # `m = (0,−4)`, sector (1,1), where `Δ_N` is constant on the whole
        # support so EVERY level above 0 is forced.  This is the same reasoning
        # as the existing "no bubbled cells at all" short-circuit in
        # `solve_level`, one notch finer: no free PARAMETERS rather than no
        # cells.  Constructive — it reads off a measured law, it does not solve.
        # (Re-measured 2026-09-26 on the multi-slot tower: the `inconsistent`
        # does not reproduce there — see `matter_multislot`, same short-circuit.)
        forced = n and not any(n <= budget[c]
                               for c in pres if c not in orbit)
        if forced:
            Ff = {c: offset.get(c, TorusRational.zero(datum))
                  for c in pres if c not in orbit}
            # D2 PROPOSES; the axioms DISPOSE.  Assemble the level and check it
            # against (★)+W1; fall through to the solve if it does not hold.
            trial = dict(Ff)
            for a in pres:
                if a in orbit:
                    Zn = Z_levels(datum, matter, a, kmax=n).get(n)
                    trial[a] = (TorusRational.zero(datum) if Zn is None
                                else (pres[a] * TorusRational.from_laurent(Zn)
                                      ).simplify())
            good, why = verify_forced_level(datum, trial)
            if verbose:
                print(f"  [level {n}: degree law forces; {why}]", flush=True)
            if not good:
                forced = False
        if not forced:
            status, Ff, _interior = solve_level(
                datum, matter, m, e, n, pad=pad, verbose=verbose,
                oq_offset=offset or None)
            if status != "unique":
                raise DivisibilityFailure(
                    f"{datum.name}: L_{{{m},{e}}} μ-level {n} not pinned by "
                    f"(★)+bar+O(𝖖) (status: {status})")

        lvl = dict(Ff)
        for a in pres:
            if a in orbit:
                Zn = Z_levels(datum, matter, a, kmax=n).get(n)
                lvl[a] = (TorusRational.zero(datum) if Zn is None
                          else (pres[a] * TorusRational.from_laurent(Zn)
                                ).simplify())
        out[n] = lvl
        for cell, f in lvl.items():
            qn = f if cell in orbit or not offset else \
                (f + offset.get(cell, TorusRational.zero(datum)) * (-1))
            if cell in orbit:
                qn = pres[cell] if n == 0 else TorusRational.zero(datum)
            Q.setdefault(cell, {})[n] = qn.simplify()

    cells: dict = {}
    for n, lvl in out.items():
        for a, f in lvl.items():
            if not f.is_zero():
                cells.setdefault(tuple(a), {})[n] = f
    check_divisibility(datum, matter, m, cells, pure=pres, strict=True)
    return out


def cells_from_rg_chart(chart: dict) -> dict:
    """Transpose `GMatterOverPure.rg_chart`'s `{level: WRQTorus}` into the
    `{atom: {level: TorusRational}}` shape the guard consumes."""
    out: dict = {}
    for k, x in chart.items():
        lvl = sum(k) if isinstance(k, tuple) else int(k)
        for atom, f in x.residuals().items():
            out.setdefault(tuple(atom), {})[lvl] = f.simplify()
    return out

def matter_theta_twist(x, k: int):
    """`T^k : f_{p,k⃗} ↦ v^{k·p̄}·f_{p,k⃗}` on a `MatterWRQTorus` — the matter lift
    of `star_bubbling.theta_twist`.

    **MEASURED (2026-07-28): the twist survives matter.**  `T^k` carries the
    canonical `L^N_{m,e}` to the canonical `L^N_{m, e + k·m̄}`, exactly as in pure
    gauge — 7/7 at SU(2)+1×2 (`m = (1,), (2,)`), SU(2)+2×2 and Sp(4)+1×4, for
    `k = 1, 2`: W1 holds, `well_formed()` returns precisely the predicted label,
    and the image **equals** the independently built chart
    (a probe in the source repository).  Confirmed by the author the same day.

    It was not obvious: the matter chart is the quotient `Q` of `F = Z·Q` and the
    dressing `Z(m)` carries `v`-dependence of its own, so the twist had to commute
    with the `Z`-division as well as with the μ-grading.  It does — the twist is a
    pure `v`-monomial per cell and the μ-levels are untouched, which is why the
    lift is level-by-level.

    Practical content: the dressed matter cones `{L^N_{m, k·m̄}}` are `T^k`-images
    of the undressed `{L^N_{m,0}}` cone and cost **no solve at all** — the first
    constructive route on the matter side (`GNAbeKAlgebra` dispatches it as
    `twist[k=…]`).

    Scope: needs `m` in the **coroot span**, since `m̄` is `cochar_to_weight(m)`,
    which honest-fails on a central-torus direction (so at `U(N)` there is no
    twist cone through the central directions — those labels are minuscule and
    already free)."""
    from matter_wrq_torus import MatterWRQTorus
    from star_bubbling import cochar_to_weight
    from weyl_torus_ring import TorusLaurent, TorusRational

    datum = x.datum
    out: dict = {}
    for p_, row in x.residuals().items():
        w = cochar_to_weight(datum, p_)
        mono = TorusRational.from_laurent(
            TorusLaurent.monomial(datum, tuple(k * c for c in w)))
        dst = out.setdefault(tuple(p_), {})
        for lev, f in row.items():
            dst[tuple(lev)] = (f * mono).simplify()
    return MatterWRQTorus(datum, x.slots, out)
