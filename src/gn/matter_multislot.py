"""`matter_multislot` — the multi-index (`M ≥ 2`) layer of the `(G, N)` constructor.

`matter_star_bubbling` works with the **total** μ-degree.  With one
hypermultiplet slot that is the whole story; with `M ≥ 2` it is a *central
specialization* (all `μ_i` set equal), and while the specialized element is
still consistent — a central specialization is an algebra homomorphism — the
**solve does not survive it**.  Measured: SU(2) `nf=2` at `m=(1,),(2,)` and U(2)
`nf=2` at `m=(0,−1),(0,−2)` all diverge from the flow, and always first at total
level 1 — precisely where the multi-index has more than one component
(`(1,0)` and `(0,1)` are distinct flavour sectors that the collapse adds).

So here the μ-level is a genuine multi-index `k⃗ ∈ Z_{≥0}^M`, one component per
irrep summand `N_i` — which is also what the flavour convention demands (one
`U(1)` per summand).

Everything else carries over unchanged in shape:

    Δ⃗_N(n)_i  =  Σ_{w ∈ wt(N_i)} max(0, −⟨n, w⟩)          (per slot)
    Z_n(μ⃗)    =  ∏_i ∏_{w: c<0} ∏_{s<|c|} (1 + μ_i 𝖖^{2s+c+1} v^w)
    F_{m'}(μ⃗) =  Z_{m'}(μ⃗) · Q_{m'}(μ⃗)
    deg Q_{m'} =  Δ⃗_N(m) − Δ⃗_N(m')                        (componentwise)

and the solve proceeds over `k⃗` in **total-degree order** (any linear extension
of the componentwise order works; total degree is the cheapest), with the
`O(𝖖)` offset `Σ_{0 ≠ i⃗ ≤ k⃗} Z[i⃗]·Q[k⃗−i⃗]` — the same identity as before, since
`Z[0⃗] = 1`.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from itertools import product

from laurent_poly import LaurentPoly
from root_datum import RootDatum
from weyl_torus_ring import TorusLaurent, TorusRational
from matter_star_bubbling import (DivisibilityFailure, divide_by_linear,
                                  matter_monomials, matter_pairing,
                                  solve_level,
                                  verify_forced_level, _weights)


__all__ = [
    "delta_N_vec",
    "Z_levels_vec",
    "divide_by_Z_vec",
    "check_divisibility_vec",
    "solve_canonical_matter_vec",
    "cells_from_rg_chart_vec",
]


def _nslots(datum, matter):
    return len(_weights(datum, matter))


def delta_N_vec(datum: RootDatum, matter, n) -> tuple:
    """`Δ⃗_N(n)` — the matter-pole count **per slot**, as Python `int`s.

    The `int` matters.  A cocharacter of a non-simply-connected global form has
    `Fraction` coordinates, so `⟨n, w⟩` comes back as a `Fraction` even when its
    value is a plain integer — and every consumer uses these entries as loop
    bounds (`range(t + 1)` over the μ-sectors), which rejects `Fraction` on type
    alone.  Since a legal matter representation of the form has all its weights in
    the form's electric lattice, `⟨n, w⟩ ∈ Z` there, so the coercion is always
    legitimate and this is what lets the μ-sector walk run at PSU(N) at all
    (measured 2026-07-28).

    Honest-fails if a count is genuinely non-integral: that means the matter is
    **not** a representation of this global form (a centre-charged weight paired
    against a fractional coweight), which `GNAbeKAlgebra` now rejects up front with
    the physics reason."""
    out = []
    for wts in _weights(datum, matter):
        tot = 0
        for w in wts:
            c = matter_pairing(datum, n, w)
            if c < 0:
                tot += -c
        itot = int(tot)
        if itot != tot:
            raise NotImplementedError(
                f"{datum.name}: Δ_N = {tot} is not an integer at cocharacter "
                f"{tuple(n)} — a matter weight carries centre charge against a "
                f"fractional coweight, i.e. the matter is not a representation "
                f"of this global form.")
        out.append(itot)
    return tuple(out)


def Z_levels_vec(datum: RootDatum, matter, n, kmax=None) -> dict:
    """`{k⃗: TorusLaurent}` — the `μ⃗^{k⃗}` coefficients of `Z_n(μ⃗)`.

    Per slot this is the elementary symmetric polynomial in that slot's matter
    monomials; across slots the coefficients multiply (the slots are
    independent `U(1)`s)."""
    M = _nslots(datum, matter)
    per_slot = [{} for _ in range(M)]
    monos = matter_monomials(datum, matter, n)
    for i in range(M):
        levels = {0: TorusLaurent.one(datum)}
        cap = None if kmax is None else kmax[i]
        for (slot, w, E) in monos:
            if slot != i:
                continue
            x = TorusLaurent.monomial(datum, tuple(w), LaurentPoly({E: 1}))
            nxt = {}
            for k, v in levels.items():
                nxt[k] = nxt.get(k, TorusLaurent.zero(datum)) + v
                if cap is None or k + 1 <= cap:
                    nxt[k + 1] = (nxt.get(k + 1, TorusLaurent.zero(datum))
                                  + v * x)
            levels = nxt
        per_slot[i] = levels
    out = {}
    for combo in product(*[sorted(s) for s in per_slot]):
        acc = TorusLaurent.one(datum)
        for i, k in enumerate(combo):
            acc = acc * per_slot[i][k]
        if not acc.is_zero():
            out[tuple(combo)] = acc
    return out


def divide_by_Z_vec(datum, matter, n, coeffs: dict):
    """Divide a cell's multivariate μ-polynomial `{k⃗: TorusRational}` by
    `Z_n(μ⃗)`, one linear factor at a time in that factor's own μ-variable.

    Returns `(quotient, ok)`; `ok` is the divisibility condition at this cell."""
    M = _nslots(datum, matter)
    work = {tuple(k): v for k, v in coeffs.items() if not v.is_zero()}
    for (slot, w, E) in matter_monomials(datum, matter, n):
        x = TorusRational.from_laurent(
            TorusLaurent(datum, {tuple(w): LaurentPoly({E: 1})}))
        # group by the other slots' indices, divide along `slot`
        groups: dict = {}
        for k, v in work.items():
            rest = k[:slot] + k[slot + 1:]
            groups.setdefault(rest, {})[k[slot]] = v
        nxt: dict = {}
        for rest, col in groups.items():
            quo, rem = divide_by_linear(datum, col, x)
            if not rem.is_zero():
                return work, False
            for j, v in quo.items():
                if v.is_zero():
                    continue
                key = rest[:slot] + (j,) + rest[slot:]
                nxt[key] = v
        work = nxt
    return {k: v for k, v in work.items() if not v.is_zero()}, True


def check_divisibility_vec(datum, matter, m, cells: dict, pure=None,
                           strict: bool = False):
    """Multi-index form of the divisibility guard.  Checks D1 (divisibility),
    D2 (`deg Q = Δ⃗_N(m) − Δ⃗_N(m')`, componentwise) and, when `pure` is given,
    D3 (`Q[0⃗] == pure`)."""
    lead = delta_N_vec(datum, matter, tuple(m))
    out = {}
    for atom, levels in cells.items():
        atom = tuple(atom)
        dn = delta_N_vec(datum, matter, atom)
        quot, ok = divide_by_Z_vec(datum, matter, atom, levels)
        deg = tuple(max((k[i] for k in quot), default=0)
                    for i in range(len(lead)))
        want = tuple(lead[i] - dn[i] for i in range(len(lead)))
        zero = (0,) * len(lead)
        is_pure = None
        if pure is not None and atom in pure:
            is_pure = (zero in quot
                       and quot[zero].simplify() == pure[atom].simplify())
        out[atom] = {"delta": dn, "quotient": quot, "degree": deg,
                     "divisible": ok, "quotient_is_pure": is_pure}
        if strict:
            if not ok:
                raise DivisibilityFailure(
                    f"cell {atom}: μ-polynomial not divisible by Z "
                    f"(Δ⃗_N={dn})")
            if deg != want:
                raise DivisibilityFailure(
                    f"cell {atom}: quotient degree {deg} != "
                    f"Δ⃗_N(m)−Δ⃗_N(m') = {want}")
    return out


def cells_from_rg_chart_vec(chart: dict) -> dict:
    """`{atom: {k⃗: TorusRational}}` from `GMatterOverPure.rg_chart` — keeping
    the multi-index instead of collapsing to total degree."""
    out: dict = {}
    for k, x in chart.items():
        kv = tuple(k)
        for atom, f in x.residuals().items():
            out.setdefault(tuple(atom), {})[kv] = f.simplify()
    return out


def solve_canonical_matter_vec(datum, matter, m, e, pad: int = 1,
                               verbose: bool = False, lines=None,
                               support=None, stats=None):
    """`L_{m,e}` for `(G, N)` with `M ≥ 2` slots, as `{k⃗: {cell: TorusRational}}`.

    Solves the μ-sectors `k⃗ ≤ Δ⃗_N(m)` in **total-degree order** (a linear
    extension of the componentwise order), each by `solve_level` with the
    `O(𝖖)` offset `Σ_{0⃗ ≠ i⃗ ≤ k⃗} Z[i⃗]·Q[k⃗−i⃗]`, then applies the multi-index
    divisibility guard in strict mode.

    `lines` is the **4d gauge group data** (`global_form.LineLattice`) and must be
    threaded in by the caller: the `μ⁰` slice is a *pure* canonical, so this
    function builds its own `PureGAbeKAlgebra`, and without `lines` that defaults
    to the **simply connected** form.  Measured consequence of omitting it
    (2026-07-28): PSU(3)'s minuscule 't Hooft line builds in the pure theory but
    honest-fails as soon as matter is added, because the inner pure algebra refuses
    the fractional coweight it was never told about.

    `support` (default `None` = the hull `conv(W·m)`, unchanged) MEASURES the
    support condition rather than assuming it (the design record,
    `conj:abeKalgebra/support`): every μ-sector is solved on that larger
    Weyl-invariant set of cells (`star_bubbling.joint_solve`, `support=`), the
    `O(𝖖)` offsets run over all of its cells, and the returned tower carries them
    — zero outside the hull wherever the support condition follows from the
    other conditions.  With `support` EVERY sector is solved: the degree law
    below is a statement about the hull's cells, and it must not decide the
    cells it is being tested against.  Measured 2026-09-26: at all 47 sectors it
    forces at U(2)+2, `m = (0,−2), (0,−3), (0,−4)`, the solve at the default
    window returns the forced answer.  `stats` (a dict, diagnostic) receives,
    per sector, `forced_on_hull` (the degree law's decision on the hull),
    `status` and the `offset` handed to the solve."""
    from pure_g_abe_kalgebra import PureGAbeKAlgebra

    m, e = tuple(m), tuple(e)
    if not datum.is_dominant_cochar(m):
        raise ValueError(
            f"{datum.name}: m={m} is not cochar-dominant "
            f"(dominant rep: {datum.dominant_cochar_rep(m)})")

    # Propagate the 4d gauge group data: without it this would default to the
    # SIMPLY CONNECTED form and refuse a fractional coweight, so a
    # non-simply-connected form that builds fine in the pure theory would
    # honest-fail as soon as matter was added (measured at PSU(3), 2026-07-28).
    P = PureGAbeKAlgebra(datum, lines=lines)
    pure = P.chart((m, e))
    pres = pure.residuals()
    orbit = {tuple(datum.act_cochar(w, m)) for w in datum.weyl_elements()}
    lead = delta_N_vec(datum, matter, m)
    M = len(lead)
    # the cells the tower lives on: the pure chart's, or those and the override's
    cells = list(pres) if support is None else list(dict.fromkeys(
        list(pres) + [tuple(p) for p in support]))

    sectors = [k for k in product(*[range(t + 1) for t in lead])]
    sectors.sort(key=lambda k: (sum(k), k))
    zero = (0,) * M
    # D2 as a per-cell, per-slot BUDGET: `deg_μ Q_{m'} = Δ⃗_N(m) − Δ⃗_N(m')`,
    # componentwise.  Where no bubbled cell can carry a free `Q[k⃗]` the sector
    # is forced, not solved — see the short-circuit in the loop.
    budget = {c: tuple(lead[t] - delta_N_vec(datum, matter, c)[t]
                       for t in range(M))
              for c in pres}

    Q: dict = {}
    out: dict = {}
    for k in sectors:
        offset: dict = {}
        if any(k):
            for cell in cells:
                if cell in orbit:
                    continue
                Zc = Z_levels_vec(datum, matter, cell, kmax=k)
                acc = TorusRational.zero(datum)
                for i in product(*[range(x + 1) for x in k]):
                    if not any(i):
                        continue
                    Zi = Zc.get(tuple(i))
                    Qj = Q.get(cell, {}).get(
                        tuple(k[t] - i[t] for t in range(M)))
                    if Zi is None or Qj is None or Qj.is_zero():
                        continue
                    acc = acc + Qj * TorusRational.from_laurent(Zi)
                offset[cell] = acc.simplify()

        # --- apply the degree law rather than re-deriving it by solve -------
        # If no bubbled cell can carry a free `Q[k⃗]` — D2 componentwise:
        # `k⃗ ≰ Δ⃗_N(m) − Δ⃗_N(m')` at every bubbled cell — then `Q[k⃗] = 0`
        # there and `f[k⃗] = offset[k⃗]` outright.  Handing such a sector to
        # `joint_solve` anyway asks it to fit unknowns that must not exist, and
        # the over-determined system reports `inconsistent`.  MEASURED at
        # U(2)+N_f=2 `m = (0,−4)`, sector (1,1): `Δ⃗_N` is constant `(4,4)`
        # across the whole support, so every sector above `0⃗` is forced; the
        # offset there was verified equal to the oracle's `Z·Q` cell by cell,
        # and with the short-circuit the label builds in 0.1 s with `Q ==
        # UNNfKAlgebra.chart` exactly — the type-A oracle, since retired (it
        # previously honest-failed outright).
        # Constructive: this reads off a measured law, it does not solve.  The
        # strict divisibility guard below still re-checks D1/D2/D3 on the
        # assembled element.
        # RE-MEASURED 2026-09-26 (the design record, the support measurement): the
        # `inconsistent` above does not reproduce on the current solver — at all
        # 47 sectors the law forces at U(2)+N_f=2, `m = (0,−2), (0,−3), (0,−4)`,
        # sector (1,1) of `m = (0,−4)` included, the solve at the default window
        # is unique and returns the forced answer.  The short-circuit stays: it
        # is cheaper, and its answer is axiom-checked below either way.
        forced = any(k) and not any(
            all(k[t] <= budget[c][t] for t in range(M))
            for c in pres if c not in orbit)
        if stats is not None:
            stats.setdefault("sectors", {})[k] = dict(forced_on_hull=bool(forced),
                                                      offset=offset)
        if support is not None:
            forced = False          # every sector solved (docstring)
        if forced:
            Ff = {c: offset.get(c, TorusRational.zero(datum))
                  for c in pres if c not in orbit}
            # D2 PROPOSES; the axioms DISPOSE.  Assemble the sector
            # and check (★)+W1; fall through to the solve if it does not hold.
            trial = dict(Ff)
            for a in pres:
                if a in orbit:
                    Zk = Z_levels_vec(datum, matter, a, kmax=k).get(k)
                    trial[a] = (TorusRational.zero(datum) if Zk is None
                                else (pres[a] * TorusRational.from_laurent(Zk)
                                      ).simplify())
            good, why = verify_forced_level(datum, trial)
            if verbose:
                print(f"  [sector {k}: degree law forces; {why}]", flush=True)
            if not good:
                forced = False
        if not forced:
            status, Ff, _interior = solve_level(
                datum, matter, m, e, k, pad=pad, verbose=verbose,
                oq_offset=offset or None, lines=lines, support=support)
            if stats is not None:
                stats["sectors"][k]["status"] = status
            if status != "unique":
                raise DivisibilityFailure(
                    f"{datum.name}: L_{{{m},{e}}} μ-sector {k} not pinned by "
                    f"(★)+bar+O(𝖖) (status: {status})")

        lvl = dict(Ff)
        for a in pres:
            if a in orbit:
                Zk = Z_levels_vec(datum, matter, a, kmax=k).get(k)
                lvl[a] = (TorusRational.zero(datum) if Zk is None
                          else (pres[a] * TorusRational.from_laurent(Zk)
                                ).simplify())
        if not forced:
            # A **SOLVED** sector is checked against the axioms too — the author's ruling,
            # 2026-07-30: *"a solved-for `L_{m,e}` should be axiom-checked if the
            # solver does not do it automatically"*.  On this path it does not:
            #
            #   * `solve_level` reaches `star_bubbling.joint_solve` directly, so it
            #     bypasses `solve_canonical`'s W1 + (★) guard — that wrapper is what
            #     carries the licence on the PURE path, and the matter path does not
            #     go through it;
            #   * `GNAbeKAlgebra._base_tower` does not certify the assembled tower
            #     either (no `well_formed`, no `certify_canonical`).
            #
            # So `status == "unique"` was the *only* acceptance a solved sector got —
            # and "unique, verified over `Z`" means the answer satisfies **its own
            # assembled rows**, not that those rows are the right ones.  A wrong
            # ansatz yielding a unique wrong answer would have passed silently.
            # (the defect was an ansatz CUTOFF, which showed up as
            # `inconsistent` rather than as a wrong answer, so this gap was not what
            # bit there — but it is the gap that would hide the same class of bug the
            # next time it produced an answer instead of a refusal.)
            #
            # The check is the FORCED branch's own — `verify_forced_level`, (★) + W1 —
            # applied to the same assembled object (`lvl` here is built exactly like
            # `trial` above), so the two paths now accept on identical grounds.
            good, why = verify_forced_level(datum, lvl)
            if verbose:
                print(f"  [sector {k}: solved; axioms "
                      f"{'OK' if good else 'VIOLATED'} — {why}]", flush=True)
            if not good:
                raise DivisibilityFailure(
                    f"{datum.name}: L_{{{m},{e}}} μ-sector {k} solved *uniquely* but "
                    f"VIOLATES the axioms: {why}.  A unique solution only means the "
                    f"answer satisfies its own assembled rows; this says the rows "
                    f"themselves are wrong (the ansatz, an offset, or the dressing) — "
                    f"so treat it as a solver defect, not as a property of the label.")
        out[k] = lvl
        for cell, f in lvl.items():
            if cell in orbit:
                qk = pres[cell] if k == zero else TorusRational.zero(datum)
            else:
                qk = (f + offset.get(cell, TorusRational.zero(datum)) * (-1)
                      ) if offset else f
            Q.setdefault(cell, {})[k] = qk.simplify()

    cells: dict = {}
    for k, lvl in out.items():
        for a, f in lvl.items():
            if not f.is_zero():
                cells.setdefault(tuple(a), {})[k] = f
    check_divisibility_vec(datum, matter, m, cells, pure=pres, strict=True)
    return out
