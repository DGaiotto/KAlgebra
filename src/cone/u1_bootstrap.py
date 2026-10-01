"""BPS-free elementary-trace generation for the u1-flavoured finite zoo, by the
orbit-reduced forward μ-bootstrap with an EXPLICIT μ-support hypothesis.

The u1 generalisation of the e6 orthonormality bootstrap (the SU3AD μ-fugacity
treatment, abelian flavour): seed traces are μ-Laurents, the cone-data Layer-1
reduction coefficients carry μ-charge, and `Tr(L) = δ_{L,1} + O(𝖖)` is imposed
on every `𝖖^{≤0}` coefficient AND μ-charge-wise.

What the sweep assumes, and where each assumption comes from:

  * **the contract axioms** — ρ²-twisted cyclicity + orthonormality (the rows);
    for the default `fold="rho"` also **ρ-equivariance of the trace**
    `Tr∘ρ = ⋆∘Tr`, in its element form on this flavour-in-coefficients cone
    tier: `ρ(c·L_w) = ⋆(c)·μ^{δ(w)}·L_{ρ(w)}` gives
    `Tr(L_{ρ(w)})(μ) = μ^{−δ(w)}·Tr(L_w)(μ⁻¹)`, so the unknowns collapse to one
    representative per ρ-orbit (e7: 90 seeds → 9); an odd ρ-orbit returns to its
    representative with `μ` flipped, which is a symmetry `Tr_rep[ν] = Tr_rep[s−ν]`
    of the representative and is imposed.  Applying the relation twice gives the
    ρ²-orbit fold `Tr(L_{ρ²w}) = μ^{δ(w)−δ(ρw)}·Tr(L_w)` (`fold="rho2"`, the
    fold this module used before; 90 → 18 on e7), which uses cyclicity only and
    is kept as the independent cross-check;
  * **the μ-support hypothesis** (MEASURED, NOT DERIVED): every seed's
    `𝖖^k` coefficient is supported on `|μ| ≤ B(k)`, `B(k) = ⌈p·k/r⌉ + s` for
    `mu_bound = (p, r, s)`; values outside are known zeros.  What it adds to
    the rows (measured 2026-09-23, both folds): with NO bound, the rows alone
    already determine every value in a finite μ-window at each order, and the
    values they determine are the ones served under the bound — a5 through
    𝖖¹² (power rows to 𝖖¹⁶): every value with |μ| ≤ 13 on every
    representative (the window is 13–17), against a true support ⌈k/3⌉ ≤ 4;
    e7 through 𝖖⁸ (power rows to 𝖖¹²): |μ| ≤ 14, against a support ⌈k/3⌉ ≤ 3.
    What no finite set of rows can do is certify that the values OUTSIDE every
    such window vanish — the values at the edge of the rows' reach are never
    determined — so with no bound the coverage guard certifies nothing.  The
    hypothesis supplies exactly that statement: no nonzero `𝖖^k μ^ν` with
    `|ν| > B(k)`.  Within the scanned family its width does not change the
    served values (every tested bound from the measured support up to 4k gives
    identical records); the default is the loosest bound tested at every
    control target (see `MU_BOUND_DEFAULT`).  The bound is recorded in the
    returned record (`record["bootstrap"]`).

The sweep (`_sweep`) runs the orders `k = 0, 1, …, K`, at each order adding the
rows whose deepest seed order is `k`.  Every seed value `(rep, order, ν)` that
is not yet DETERMINED enters every row it appears in as an unknown, at every
order `≤ k`; the elimination is incremental exact Gauss–Jordan, so a row that
still involves undetermined values stays in the system for the later orders.
A value is determined when it is fixed by the rows, and only then.  Until
2026-09-23 the sweep made a value an unknown only at a row's deepest order and
read every other never-solved value as 0: on e7 four ρ²-representatives
(`mg3, mg4, mg21, mg27`) were silently zeroed and the sweep ended
"inconsistent at k=2".

Guards (all strict):
  * a row whose unknowns are all determined and whose right-hand side is not 0
    is an inconsistency → `_BootstrapUnavailable("… inconsistent …")`; so is
    an orthonormality entry `I_aa = 1 + O(𝖖)` whose reduction is 0 or has no
    `𝖖^{≤0}` term (its `𝖖⁰` row reads `0 = 1`, and no sweep order forms it);
  * a determined non-integer value → `_BootstrapUnavailable("… non-integer …")`;
  * a nonzero reduction term on a label that is neither a seed nor the
    identity → `_BootstrapUnavailable` (its trace is never read as 0);
  * coverage: a `(rep, order ≤ K)` whose in-support values are not ALL
    determined is reported unpinned and nothing is served
    (`_BootstrapUnavailable("… unpinned …")`).  The orders above the served
    depth are solved only to help pin the served ones and are not returned;
    a solve depth below the served depth (`margin < 0`, `need > K`) is a
    `ValueError`, since an order that is never solved cannot be certified.

Tr(1) is the only non-reduction input (the Nahm sum on the BPS spec, spine-free).
The frozen a3 and a5 tables this route's predecessor was once validated against
were removed on 2026-09-23 (the a5 one was wrong from 𝖖¹⁴ in its orbit rows);
the route is now checked against the independent ungauged route and the a3
closed form.

`generate_u1(short_id, K)` returns the standard frozen-table record (plus a
`"bootstrap"` entry recording the hypothesis, the fold and the certificate); it
is the u1 path of `elem_traces.generate`.
"""
from __future__ import annotations

import time
from fractions import Fraction

from elem_traces import (
    _standalone_algebra, _series_to_data, _vacuum_rps, fold_policy,
    _BootstrapUnavailable,
)
from regen import _load_standalone, REGEN_SPECS


# ---------------------------------------------------------------------------
# The μ-support hypothesis
# ---------------------------------------------------------------------------

#: `(p, r, s)`: every seed's 𝖖^k coefficient is supported on |μ| ≤ ⌈p·k/r⌉ + s.
#: A HYPOTHESIS, measured, not derived.  Measured supports of the true seeds:
#: a3 max|μ| ≤ ⌈k/2⌉, with equality at odd k (the a3 seeds vanish at even k;
#: the closed-form characters, through 𝖖²⁰), a5 ⌈k/3⌉ exactly through 𝖖⁵²
#: (24 seeds, the ungauged route `ungauge_u1a1aodd(2)`), a7 ⌈k/4⌉ through 𝖖²⁴
#: (65 seeds, `ungauge_u1a1aodd(3)`), e7 ⌈k/3⌉ on every order this bootstrap
#: has pinned (through 𝖖¹⁴).  Scan (2026-09-23): every bound tested from the
#: measured support up to |μ| ≤ 4k pins every seed at every positive-control
#: target (a3 𝖖²⁰, a5 𝖖¹² and 𝖖²⁰, a7 𝖖⁸ and 𝖖¹⁴), with identical values
#: (|μ| ≤ 16k too, at a5 𝖖¹²); a bound below the measured support (a3 ⌈k/3⌉,
#: a5 and e7 ⌈k/4⌉, a7 ⌈k/5⌉) is detected as an inconsistency at the first
#: order it cuts.  The width does not change the served values because the
#: rows alone already determine them (module docstring: with no bound at all,
#: a5's rows determine every value with |μ| ≤ 13 through 𝖖¹²); the bound only
#: certifies that the values beyond the rows' reach vanish.  The default is
#: the loosest bound tested at every control target, |μ| ≤ 4k; its cost over
#: |μ| ≤ k is the sweep's (linear in the width: e7 at 𝖖¹² 43 s vs 11 s per
#: sweep, against ~500 s of cone-data reductions).
MU_BOUND_DEFAULT = (4, 1, 0)


def mu_bound_value(mu_bound, k):
    """B(k) for `mu_bound = (p, r, s)`: ⌈p·k/r⌉ + s."""
    p, r, s = mu_bound
    return -((-p * k) // r) + s


def mu_bound_text(mu_bound):
    if mu_bound is None:
        return "none (unbounded: nothing can be certified)"
    p, r, s = mu_bound
    pk = "k" if p == 1 else f"{p}k"
    core = pk if r == 1 else f"ceil({pk}/{r})"
    return f"|mu| <= {core}" + (f" + {s}" if s > 0 else f" - {-s}" if s < 0 else "")


# ---------------------------------------------------------------------------
# μ-Laurent (qmu) helpers
# ---------------------------------------------------------------------------

def coeff_to_qmu(c):
    """Reduction coefficient -> {q_exp: {mu: int}}."""
    if isinstance(c, dict):                         # already {q:{mu:int}}
        return c
    if hasattr(c, "coeffs"):                        # RLaurent: {q: RElement}
        out = {}
        for q, re in c.coeffs.items():
            d = {(mu[0] if isinstance(mu, tuple) else mu): int(v)
                 for mu, v in re.terms.items() if v}
            if d:
                out[q] = d
        return out
    return {q: {0: int(v)} for q, v in c._coeffs.items() if v}   # LaurentPoly


def frozen_qmu(entry):
    """Frozen u1 datum {q:{(f,):c}} -> {q:{f:c}}."""
    out = {}
    for q, terms in entry.items():
        d = {(k[0] if isinstance(k, tuple) else k): int(v)
             for k, v in terms.items() if v}
        if d:
            out[int(q)] = d
    return out


def _pair_poly_u1(A, a, b, delta_table):
    """δ-honest orthonormality pair I_{a,b}=Tr(ρ(L_a)·L_b) reduced to seeds, as
    {seed:{q:{mu:int}}}.  Since ρ(L_a)=μ^{δ(a)}·L_{ρ(a)} (element-level), the
    honest pair is μ^{δ(a)}·(label-level _pair_poly), δ(a)=Σ_{(i,p)∈a} δ(i)·p."""
    from trace_uniqueness_proofs import _pair_poly
    da = sum((delta_table.get(i) or (0,))[0] * p for i, p in a)
    out = {}
    for s, co in _pair_poly(A, a, b).items():
        qmu = coeff_to_qmu(co)
        if da:
            qmu = {q: {mu + da: c for mu, c in mud.items()}
                   for q, mud in qmu.items()}
        out[s] = qmu
    return out


def _to_pool(pool, reduction, delta, ident, pos):
    """Append one reduced trace `{label: coefficient}` to `pool` as an entry
    `(R, emin, mu_lo, mu_hi, delta)`, `R = {seed position or "id": {q: {μ: c}}}`
    (`delta`: an orthonormality entry `I_aa = 1 + O(𝖖)`).  Raises
    `_BootstrapUnavailable` on a nonzero term on a label that is neither a
    seed nor the identity (its trace is not an unknown of the sweep, and must
    not be read as 0), and on an orthonormality entry that reduces to 0."""
    R = {}
    for sl, c in reduction.items():
        qmu = coeff_to_qmu(c)
        if not qmu:
            continue
        key = "id" if sl == ident else pos.get(sl)
        if key is None:
            raise _BootstrapUnavailable(
                f"reduction term on {sl!r}, which is neither a seed nor the "
                f"identity")
        R[key] = qmu
    if not R and delta:
        raise _BootstrapUnavailable(
            "inconsistent: an orthonormality entry I_aa = 1 + O(𝖖) reduces "
            "to 0")
    if R:
        emin = min(e for r in R.values() for e in r)
        mu_lo = min((mu for r in R.values() for mud in r.values()
                     for mu in mud), default=0)
        mu_hi = max((mu for r in R.values() for mud in r.values()
                     for mu in mud), default=0)
        pool.append((R, emin, mu_lo, mu_hi, delta))


# ---------------------------------------------------------------------------
# Folds: seed -> (representative, shift, flip), Tr_s[μ] = Tr_rep[flip·(μ − shift)]
# ---------------------------------------------------------------------------

def _orbit_map(short_id, idxs, pos_of_idx):
    """ρ²-orbit fold: seed_pos -> (rep_pos, Δ) with Tr(L_s)=μ^Δ·Tr(L_rep)."""
    mod, pfx = _load_standalone(short_id)
    perm = {int(k): int(v) for k, v in getattr(mod, f"{pfx}_RHO_PERM").items()}
    dt = getattr(mod, f"{pfx}_RHO_DELTA", {}) or {}
    def rho(i):
        return perm.get(i, i)
    def dlt(i):
        v = dt.get(i)
        return v[0] if v else 0
    def rho2(i):
        return rho(rho(i))
    smap = {}
    seen = set()
    for idx in idxs:
        if idx in seen:
            continue
        # u1 (fold='none'): walk the ρ²-orbit.  trivial-R (fold='rho2'): the
        # seeds are already orbit reps, so ρ²(seed) ∉ the seed set and each
        # orbit collapses to a singleton (identity map) — the forward solver
        # then runs directly over the folded seeds (e6, e8).
        orbit, j = [], idx
        while j in pos_of_idx and j not in seen:
            orbit.append(j); seen.add(j); j = rho2(j)
        rep = min(orbit)
        x, acc = rep, 0
        for _ in range(len(orbit)):
            smap[pos_of_idx[x]] = (pos_of_idx[rep], acc)
            acc += dlt(x) - dlt(rho(x))
            x = rho2(x)
        if x == rep and acc != 0:
            raise _BootstrapUnavailable(
                f"{short_id}: ρ²-orbit of mg{rep} closes with μ^{acc} ≠ 1")
    return smap


def _fold_map(short_id, idxs, pos_of_idx, fold):
    """`(smap, sym)`: `smap[seed_pos] = (rep_pos, shift, flip)` with
    `Tr_seed[μ] = Tr_rep[flip·(μ − shift)]`; `sym[rep_pos] = s` where the
    representative's own ρ-orbit imposes `Tr_rep[ν] = Tr_rep[s − ν]`.

    `fold="rho2"`: the ρ²-orbits (cyclicity only; flip always +1, sym empty).
    `fold="rho"`: the ρ-orbits, from ρ-equivariance of the trace in its
    element form (module docstring): `Tr_{ρx}[μ] = Tr_x[−μ − δ(x)]`."""
    if fold == "rho2":
        return ({s: (r, d, 1) for s, (r, d) in
                 _orbit_map(short_id, idxs, pos_of_idx).items()}, {})
    if fold != "rho":
        raise ValueError(f"fold must be 'rho' or 'rho2', not {fold!r}")
    mod, pfx = _load_standalone(short_id)
    perm = {int(k): int(v) for k, v in getattr(mod, f"{pfx}_RHO_PERM").items()}
    dt = getattr(mod, f"{pfx}_RHO_DELTA", {}) or {}
    def dlt(i):
        v = dt.get(i)
        return v[0] if v else 0
    smap, sym, seen = {}, {}, set()
    for i0 in sorted(idxs):
        if i0 in seen:
            continue
        x, sh, fl = i0, 0, 1
        while True:
            smap[pos_of_idx[x]] = (pos_of_idx[i0], sh, fl)
            seen.add(x)
            sh, fl = -dlt(x) - sh, -fl
            x = perm.get(x, x)
            if x == i0:
                break
            if x not in pos_of_idx:
                raise _BootstrapUnavailable(
                    f"{short_id}: ρ(mg…) = mg{x} is not a seed")
        if fl == -1:
            sym[pos_of_idx[i0]] = sh
        elif sh != 0:
            raise _BootstrapUnavailable(
                f"{short_id}: ρ-orbit of mg{i0} closes with μ^{sh} ≠ 1")
    return smap, sym


# ---------------------------------------------------------------------------
# Incremental exact Gauss–Jordan elimination
# ---------------------------------------------------------------------------

class _GaussJordan:
    """Rows are added one at a time; every pivot is kept expressed in the FREE
    (non-pivot) unknowns only, so an unknown is determined iff it is a pivot
    with an empty expression — an intrinsic property of the row space, not of
    the pivot order.  `add` returns False on an inconsistent row."""

    def __init__(self):
        self.piv = {}      # u -> [expr {free v: Fraction}, const Fraction]
        self.occ = {}      # free v -> set of pivots whose expr contains v
        self.nrows = 0

    def add(self, co, rhs):
        self.nrows += 1
        row, r = {}, Fraction(rhs)
        for u, c in co.items():
            p = self.piv.get(u)
            if p is None:
                row[u] = row.get(u, 0) + c
            else:
                r -= c * p[1]
                for v, cv in p[0].items():
                    row[v] = row.get(v, 0) + c * cv
        row = {v: c for v, c in row.items() if c}
        if not row:
            return r == 0
        v = min(row, key=lambda w: (len(self.occ.get(w, ())),
                                    abs(row[w]) != 1))
        cv = row.pop(v)
        expr = {w: Fraction(-c) / cv for w, c in row.items()}
        const = r / cv
        for u in self.occ.pop(v, ()):
            e = self.piv[u][0]
            c = e.pop(v)
            self.piv[u][1] += c * const
            for w, cw in expr.items():
                nw = e.get(w, 0) + c * cw
                if nw:
                    if w not in e:
                        self.occ.setdefault(w, set()).add(u)
                    e[w] = nw
                elif w in e:
                    del e[w]
                    self.occ[w].discard(u)
        self.piv[v] = [expr, const]
        for w in expr:
            self.occ.setdefault(w, set()).add(v)
        return True

    def value(self, u):
        """The determined value of `u`, or None."""
        p = self.piv.get(u)
        return p[1] if (p is not None and not p[0]) else None


# ---------------------------------------------------------------------------
# The forward sweep
# ---------------------------------------------------------------------------

def _sweep(pool, fold, Tr1, K, *, mu_bound=MU_BOUND_DEFAULT, need=None,
           verbose=False):
    """Forward sweep over the orders 0..K with every undetermined seed value an
    unknown (module docstring).

    `fold = (smap, sym)` from `_fold_map`; `mu_bound` the support hypothesis
    (`None`: no bound — each row is taken over the μ-window of the pre-fix
    sweep, `k + 6` either side, every value it touches is an unknown, and
    nothing can be certified).  `need` (default `K`, at most `K`): the orders
    through which coverage is checked and values are returned; the orders
    `need < k ≤ K` are solved only to help pin those.

    Returns a dict: `values[rep][order][ν]` (determined nonzero values at the
    orders ≤ `need`, both members of a `sym` pair), `unpinned` {rep: [orders
    ≤ need not fully determined]}, and the certificate counts.  Raises
    `_BootstrapUnavailable` on an inconsistent row (including an
    orthonormality entry with no `𝖖^{≤0}` term) or a non-integer determined
    value, and `ValueError` if `need > K`."""
    smap, sym = fold
    if need is None:
        need = K
    if need > K:
        raise ValueError(
            f"need = {need} > K = {K}: the orders above the solve depth are "
            f"never solved, so they cannot be certified")
    reps = sorted({r for r, _, _ in smap.values()})
    members = {}
    for s, (r, sh, fl) in smap.items():
        members.setdefault(r, []).append(fl * sh)
    supp_cache = {}

    def interval(rep, k):
        """ν-interval of rep at order k: the bound imposed on EVERY seed of the
        orbit, |fl·ν + sh| ≤ B(k)  ⇔  −B − fl·sh ≤ ν ≤ B − fl·sh."""
        key = (rep, k)
        if key not in supp_cache:
            B = mu_bound_value(mu_bound, k)
            ms = members[rep]
            lo, hi = -B - min(ms), B - max(ms)
            if rep in sym:                       # ν and s−ν both in support
                s = sym[rep]
                lo, hi = max(lo, s - hi), min(hi, s - lo)
            supp_cache[key] = (lo, hi)
        return supp_cache[key]

    def unknown(rep, k, nu):
        """Canonical unknown for Tr_rep[𝖖^k μ^ν], None if out of support."""
        if mu_bound is not None:
            lo, hi = interval(rep, k)
            if not lo <= nu <= hi:
                return None
        if rep in sym:
            nu = min(nu, sym[rep] - nu)
        return (rep, k, nu)

    entries = []
    for R, emin, mu_lo, mu_hi, delta in pool:
        if delta and emin > 0:
            # every term is O(𝖖^emin): the 𝖖⁰ row reads 0 = 1, and the loop
            # below (rows at 𝖖^{k + emin ≤ 0} only) would never form it
            raise _BootstrapUnavailable(
                "inconsistent: an orthonormality entry I_aa = 1 + O(𝖖) has "
                "no 𝖖^{≤0} term")
        terms = [(key, e, mu1, c) for key, qmu in R.items()
                 for e, mud in qmu.items() for mu1, c in mud.items() if c]
        entries.append((terms, emin, mu_lo, mu_hi, delta))

    GJ = _GaussJordan()
    touched = set()
    t0 = time.time()
    for k in range(0, K + 1):
        nrow0 = GJ.nrows
        for terms, emin, mu_lo, mu_hi, delta in entries:
            m = k + emin
            if m > 0:
                continue
            if mu_bound is None:
                W = k + 6
                cands = set(range(mu_lo - W, mu_hi + W + 1))
            else:
                cands = set()
                for key, e, mu1, c in terms:
                    ix = m - e
                    if key == "id":
                        if ix >= 0:
                            cands.update(mu1 + mu2 for mu2 in Tr1.get(ix, ()))
                    elif ix >= 1:
                        B = mu_bound_value(mu_bound, ix)
                        cands.update(range(mu1 - B, mu1 + B + 1))
            if delta and m == 0:
                cands.add(0)
            for mt in cands:
                co = {}
                rhs = 1 if (delta and m == 0 and mt == 0) else 0
                for key, e, mu1, c in terms:
                    ix = m - e
                    mu2 = mt - mu1
                    if key == "id":
                        if ix >= 0:
                            rhs -= c * Tr1.get(ix, {}).get(mu2, 0)
                    elif ix >= 1:
                        rep, sh, fl = smap[key]
                        u = unknown(rep, ix, fl * (mu2 - sh))
                        if u is not None:
                            co[u] = co.get(u, 0) + c
                co = {u: c for u, c in co.items() if c}
                if not co and not rhs:
                    continue
                touched.update(co)
                if not GJ.add(co, rhs):
                    raise _BootstrapUnavailable(
                        f"inconsistent at order {k} (μ-support hypothesis "
                        f"{mu_bound_text(mu_bound)}): a row with every value "
                        f"determined has nonzero right-hand side")
        if verbose:
            ndet = sum(1 for p in GJ.piv.values() if not p[0])
            print(f"    order {k}: +{GJ.nrows - nrow0} rows; "
                  f"{len(touched)} unknowns touched, {ndet} determined "
                  f"[{time.time() - t0:.1f}s]", flush=True)

    # values and the coverage guard
    values, unpinned = {}, {}
    for u, (expr, const) in GJ.piv.items():
        if expr:
            continue
        if const.denominator != 1:
            raise _BootstrapUnavailable(
                f"non-integer determined value {u} = {const}")
    for rep in reps:
        for k in range(1, need + 1):
            if mu_bound is None:
                us = sorted({u for u in touched if u[0] == rep and u[1] == k})
            else:
                lo, hi = interval(rep, k)
                us = [unknown(rep, k, nu) for nu in range(lo, hi + 1)]
            ok = True
            for u in us:
                v = GJ.value(u)
                if v is None:
                    ok = False
                    continue
                if v:
                    values.setdefault(rep, {}).setdefault(k, {})[u[2]] = int(v)
                    if rep in sym:
                        values[rep][k][sym[rep] - u[2]] = int(v)
            if mu_bound is None or not ok:
                unpinned.setdefault(rep, []).append(k)
    ndet = sum(1 for p in GJ.piv.values() if not p[0])
    return {
        "values": values,
        "unpinned": unpinned,
        "rows": GJ.nrows,
        "unknowns_touched": len(touched),
        "determined": ndet,
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Pool + driver
# ---------------------------------------------------------------------------

def _power_pool(A, idxs, ident, pos, K, amax=6):
    """Single-generator powers `((i, a),)`, a = 2..amax, per seed until the
    reduction's reach (−min 𝖖-exponent) is ≥ K."""
    from trace_uniqueness_proofs import seed_reduction
    pool = []
    for idx in idxs:
        for a in range(2, amax + 1):
            try:
                red = seed_reduction(A, ((idx, a),))
            except Exception:
                break
            _to_pool(pool, red, False, ident, pos)
            reach = -min((e for sl in red.values()
                          for e in coeff_to_qmu(sl)), default=0)
            if reach >= K:
                break
    return pool


def _add_pairs(pool, A, reps_idx, idxs, delta_table, ident, pos):
    """δ-honest orthonormality pairs `I_{a,b} = δ_{a,b} + O(𝖖)` for the given
    representatives.  A pair whose reduction raises adds no row (fewer rows
    can only leave values undetermined); the guards of `_to_pool` are NOT
    caught here, so a violated orthonormality entry or a non-seed label
    reaches the caller."""
    for ridx in reps_idx:
        for a in (1, 2, 3):
            la = ((ridx, a),)
            for lb in ([((ridx, b),) for b in (1, 2, 3)]
                       + [((jj, 2),) for jj in idxs]):
                try:
                    red = _pair_poly_u1(A, la, lb, delta_table)
                except Exception:
                    continue
                _to_pool(pool, red, la == lb, ident, pos)


def forward_bootstrap_u1(short_id, K, Tr1, *, need=None, fold="rho",
                         mu_bound=MU_BOUND_DEFAULT, verbose=False):
    """Forward solver: the single-generator powers, then orthonormality pairs
    for the representatives still unpinned through `need` (default `K`), one
    representative at a time.

    Returns `(seedlabs, per_seed, info)`: `per_seed[pos] = {k: {μ: int}}`
    (orders ≤ `need` only, determined values only — an order listed in
    `info["unpinned"]` is partial) and `info` the last sweep's certificate
    plus `unpinned_seeds` (seed indices whose representative is unpinned
    through `need`).  Raises `_BootstrapUnavailable` on an inconsistency or a
    non-integer value, and `ValueError` if `need > K`."""
    from trace_uniqueness_proofs import seed_set
    if need is None:
        need = K
    if need > K:
        raise ValueError(
            f"need = {need} > K = {K}: the orders above the solve depth are "
            f"never solved, so they cannot be certified")
    A = _standalone_algebra(short_id)
    ident = A.identity()
    seedlabs = [s for s in seed_set(A) if s != ident]
    pos = {sl: p for p, sl in enumerate(seedlabs)}
    idxs = [sl[0][0] for sl in seedlabs]
    n = len(seedlabs)
    pos_of_idx = {idxs[p]: p for p in range(n)}
    fmap = _fold_map(short_id, idxs, pos_of_idx, fold)
    _mod, _pfx = _load_standalone(short_id)
    delta_table = getattr(_mod, f"{_pfx}_RHO_DELTA", {}) or {}
    reps = sorted({r for r, _, _ in fmap[0].values()})
    if verbose:
        print(f"  fold={fold}: {len(reps)} representatives / {n} seeds; "
              f"μ-support hypothesis {mu_bound_text(mu_bound)}", flush=True)

    t0 = time.time()
    pool = _power_pool(A, idxs, ident, pos, K)
    t_pool = time.time() - t0
    if verbose:
        print(f"  single-generator powers: {len(pool)} labels "
              f"({t_pool:.1f}s)", flush=True)
    res = _sweep(pool, fmap, Tr1, K, mu_bound=mu_bound, need=need,
                 verbose=verbose)
    # Pairs one representative at a time, the earliest-unpinned first, with a
    # re-sweep after each: one representative's pairs often pin others too
    # (e7 at 𝖖¹⁴: mg2's pairs pin mg0, mg1, mg3), and pair reductions are
    # expensive (all nine e7 representatives: > 20 CPU-min, > 3 GB).  Adding
    # rows never changes a determined value, so this changes cost only.
    t_pairs, npairs, paired = 0.0, 0, set()
    while res["unpinned"] and mu_bound is not None:
        todo = [r for r in sorted(res["unpinned"],
                                  key=lambda r: (min(res["unpinned"][r]), r))
                if r not in paired]
        if not todo:
            break
        r = todo[0]
        paired.add(r)
        if verbose:
            print(f"  {len(res['unpinned'])}/{len(reps)} representatives "
                  f"unpinned (orders {res['unpinned']}); adding pairs for "
                  f"mg{idxs[r]}", flush=True)
        t0 = time.time()
        n0 = len(pool)
        _add_pairs(pool, A, [idxs[r]], idxs, delta_table, ident, pos)
        t_pairs += time.time() - t0
        npairs += len(pool) - n0
        if verbose:
            print(f"  pool {len(pool)} (+{len(pool) - n0} pairs; "
                  f"{time.time() - t0:.1f}s)", flush=True)
        res = _sweep(pool, fmap, Tr1, K, mu_bound=mu_bound, need=need,
                     verbose=verbose)

    smap = fmap[0]
    out = [dict() for _ in range(n)]
    for s in range(n):
        rep, sh, fl = smap[s]
        for k, mud in res["values"].get(rep, {}).items():
            for nu, v in mud.items():
                out[s].setdefault(k, {})[fl * nu + sh] = v
    info = dict(res)
    info.update(pool=len(pool), pairs=npairs, pool_seconds=t_pool,
                pair_seconds=t_pairs, representatives=[idxs[r] for r in reps],
                paired=sorted(idxs[r] for r in paired),
                unpinned={idxs[r]: ks for r, ks in res["unpinned"].items()},
                unpinned_seeds=sorted(idxs[s] for s in range(n)
                                      if smap[s][0] in res["unpinned"]))
    del info["values"]
    return seedlabs, out, info


def generate_u1(short_id, K, *, margin=4, fold="rho",
                mu_bound=MU_BOUND_DEFAULT, verbose=False):
    """BPS-free elementary-trace record for a u1 entry (the u1 path of
    `elem_traces.generate`).  Tr(1) from the Nahm sum; the seeds from the
    forward bootstrap, solved to `K + margin` (the deeper rows help pin the
    orders ≤ K) and served through `K`.

    Raises `_BootstrapUnavailable` if the sweep is inconsistent or non-integer
    under the stated μ-support hypothesis, or if any seed is unpinned through
    `K` (the coverage guard) — nothing is served in those cases — and
    `ValueError` if `margin < 0` (the orders above `K + margin` would be
    served unsolved).  The record's `"bootstrap"` entry states the
    hypothesis, the fold and the certificate."""
    if REGEN_SPECS[short_id][2] != "u1":
        raise _BootstrapUnavailable(f"{short_id}: flavour is not u1")
    if mu_bound is None:
        raise _BootstrapUnavailable(
            f"{short_id}: no μ-support hypothesis — nothing can be certified")
    if margin < 0:
        raise ValueError(
            f"{short_id}: margin = {margin} < 0 — the orders above the solve "
            f"depth K + margin would be served unsolved")
    Ki = K + margin
    if verbose:
        print(f"[{short_id}] u1 bootstrap: Tr(1) via Nahm-sum (spec) at K={Ki} "
              f"(spine-free) ...", flush=True)
    ident_full = _series_to_data(short_id, _vacuum_rps(short_id, Ki))
    Tr1 = frozen_qmu(ident_full)
    try:
        seedlabs, per_seed, info = forward_bootstrap_u1(
            short_id, Ki, Tr1, need=K, fold=fold, mu_bound=mu_bound,
            verbose=verbose)
    except _BootstrapUnavailable as e:
        raise _BootstrapUnavailable(f"{short_id}: {e}") from None
    if info["unpinned"]:
        raise _BootstrapUnavailable(
            f"{short_id}: {len(info['unpinned_seeds'])} seeds unpinned "
            f"through 𝖖^{K} under {mu_bound_text(mu_bound)} "
            f"(representative: orders) {info['unpinned']}")
    idxs = [sl[0][0] for sl in seedlabs]
    orbits = {}
    for j in range(len(idxs)):
        orbits[idxs[j]] = {q: {(mu,): c for mu, c in mud.items() if c}
                           for q, mud in per_seed[j].items()
                           if q <= K and any(mud.values())}
    if verbose:
        print(f"[{short_id}] u1 bootstrap pinned all {len(idxs)} seeds through "
              f"𝖖^{K} ({info['rows']} rows, {info['determined']} determined "
              f"values, consistent)", flush=True)
    return {
        "K": K,
        "flavor": "u1",
        "fold": fold_policy(short_id),
        "identity": {q: t for q, t in ident_full.items() if q <= K},
        "orbits": orbits,
        "bootstrap": {
            "mu_support_hypothesis": mu_bound_text(mu_bound),
            "mu_bound": tuple(mu_bound),
            "trace_fold": fold,
            "solved_through": Ki,
            "rows": info["rows"],
            "determined": info["determined"],
            "pool": info["pool"],
            "pairs": info["pairs"],
            "pairs_for": info["paired"],
        },
    }
