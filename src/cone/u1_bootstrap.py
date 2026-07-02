"""BPS-free elementary-trace generation for the u1-flavoured finite zoo, by the
ρ²-orbit-reduced forward-triangular μ-bootstrap.

The u1 generalisation of the e6 orthonormality bootstrap (the SU3AD μ-fugacity
treatment, abelian flavour): seed traces are μ-Laurents, the cone-data Layer-1
reduction coefficients carry μ-charge, and `Tr(L)=O(q)` is imposed
q^{≤0}-coefficient-wise AND μ-charge-wise.  Two ideas make it scale:

  * **ρ²-orbit reduction** — within a ρ²-orbit the seed traces are μ-power
    multiples of one rep (`Tr(L_{ρ²i})=μ^{δ(i)−δ(ρi)}·Tr(L_i)`, from
    ρ²-invariance + the `_rho_delta` unit), so the unknowns collapse to one rep
    per orbit and the rest are recovered by μ-shift (e7: 90 seeds → 18 reps).
  * **forward-triangular sweep** — order k=1..K solving only the small per-order
    frontier system; the deep single-mult-gen identity-pairings pin most reps,
    the few free reps get the δ-honest orthonormality pairs.

Tr(1) (the vacuum character, served spine-free) is the only external input —
a per-seed BPS engine would in any case be infeasible on the E-series.
Validated BPS-free vs the frozen tables: a3 6/6, a5 24/24; e7 (for which no
frozen reference table exists) generated with all 90 seeds pinned and an
over-determined-consistent certificate.

`generate_u1(short_id, K)` returns the standard frozen-table record; it is the
u1 path of `elem_traces.generate`.
"""
from __future__ import annotations

import time

from elem_traces import (
    _standalone_algebra, _solve_full, _bps_oracle, _series_to_data,
    _vacuum_rps, fold_policy, _BootstrapUnavailable,
)
from regen import _load_standalone, REGEN_SPECS


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
    R = {}
    for sl, c in reduction.items():
        key = "id" if sl == ident else pos.get(sl)
        if key is None:
            continue
        qmu = coeff_to_qmu(c)
        if qmu:
            R[key] = qmu
    if R:
        emin = min(e for r in R.values() for e in r)
        mu_lo = min((mu for r in R.values() for mud in r.values()
                     for mu in mud), default=0)
        mu_hi = max((mu for r in R.values() for mud in r.values()
                     for mu in mud), default=0)
        pool.append((R, emin, mu_lo, mu_hi, delta))


def _orbit_map(short_id, idxs, pos_of_idx):
    """seed_pos -> (rep_pos, Δ) with Tr(L_s)=μ^Δ·Tr(L_rep)."""
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
    return smap


def _sweep(pool, seed_map, Tr1, K, strict=True):
    """Forward k-sweep, unknowns reduced to ρ²-orbit reps."""
    Tr = {}

    def known(s, ix, mu):
        rep, Dl = seed_map[s]
        return Tr.get(rep, {}).get(ix, {}).get(mu - Dl, 0) if ix >= 1 else 0
    free_reps = set()
    for k in range(1, K + 1):
        eqs = []
        for R, emin, mu_lo, mu_hi, delta in pool:
            m = k + emin
            if m > 0:
                continue
            W = k + 6
            for mu_tot in range(mu_lo - W, mu_hi + W + 1):
                co = {}
                rhs = 1 if (delta and m == 0 and mu_tot == 0) else 0
                for key, qmu in R.items():
                    for e, mud in qmu.items():
                        ix = m - e
                        for mu1, c in mud.items():
                            mu2 = mu_tot - mu1
                            if key == "id":
                                if ix >= 0:
                                    rhs -= c * Tr1.get(ix, {}).get(mu2, 0)
                            elif ix == k:
                                rep, Dl = seed_map[key]
                                co[(rep, mu2 - Dl)] = \
                                    co.get((rep, mu2 - Dl), 0) + c
                            elif 1 <= ix < k:
                                rhs -= c * known(key, ix, mu2)
                if co or rhs:
                    eqs.append((co, rhs))
        unk = sorted({u for co, _ in eqs for u in co})
        sol, free, consistent = _solve_full(eqs, unk)
        if not consistent:
            if strict:
                return None, f"inconsistent at k={k}"
            free_reps |= {u[0] for co, _ in eqs for u in co}
            continue
        for (rep, mu), v in (sol or {}).items():
            if v != 0:
                if v.denominator != 1:
                    return None, f"non-integer at k={k}: {v}"
                Tr.setdefault(rep, {}).setdefault(k, {})[mu] = int(v)
        free_reps |= {rep for (rep, mu) in free}
    return Tr, free_reps


def forward_bootstrap_u1(short_id, K, Tr1, *, verbose=False):
    """Two-phase orbit-reduced forward solver; returns (seedlabs, (Tr, free), status)."""
    from trace_uniqueness_proofs import seed_set, seed_reduction
    A = _standalone_algebra(short_id)
    ident = A.identity()
    seedlabs = [s for s in seed_set(A) if s != ident]
    pos = {sl: p for p, sl in enumerate(seedlabs)}
    idxs = [sl[0][0] for sl in seedlabs]
    n = len(seedlabs)
    pos_of_idx = {idxs[p]: p for p in range(n)}
    seed_map = _orbit_map(short_id, idxs, pos_of_idx)
    _mod, _pfx = _load_standalone(short_id)
    delta_table = getattr(_mod, f"{_pfx}_RHO_DELTA", {}) or {}
    reps = sorted({rep for rep, _ in seed_map.values()})
    if verbose:
        print(f"  {len(reps)} ρ²-orbit reps / {n} seeds", flush=True)

    pool = []
    t0 = time.time()
    for idx in idxs:
        for a in range(2, 7):
            try:
                red = seed_reduction(A, ((idx, a),))
            except Exception:
                break
            _to_pool(pool, red, False, ident, pos)
            reach = -min((e for sl in red.values()
                          for e in coeff_to_qmu(sl)), default=0)
            if reach >= K:
                break
    if verbose:
        print(f"  deep pool: {len(pool)} labels ({time.time()-t0:.1f}s)",
              flush=True)
    _, free = _sweep(pool, seed_map, Tr1, K, strict=False)

    if free:
        if verbose:
            print(f"  {len(free)}/{len(reps)} reps need pairs", flush=True)
        t0 = time.time()
        for rep in sorted(free):
            ridx = idxs[rep]
            for a in (1, 2, 3):
                for b in (1, 2, 3):
                    try:
                        _to_pool(pool, _pair_poly_u1(A, ((ridx, a),),
                                                     ((ridx, b),), delta_table),
                                 a == b, ident, pos)
                    except Exception:
                        pass
                for jj in idxs:
                    try:
                        _to_pool(pool, _pair_poly_u1(A, ((ridx, a),),
                                                     ((jj, 2),), delta_table),
                                 (ridx, a) == (jj, 2), ident, pos)
                    except Exception:
                        pass
        if verbose:
            print(f"  pool {len(pool)} (+pairs; {time.time()-t0:.1f}s)",
                  flush=True)

    Tr, free = _sweep(pool, seed_map, Tr1, K, strict=True)
    if Tr is None:
        return seedlabs, None, free

    out = [dict() for _ in range(n)]
    for s in range(n):
        rep, Dl = seed_map[s]
        for k, mud in Tr.get(rep, {}).items():
            for mu, v in mud.items():
                if v:
                    out[s].setdefault(k, {})[mu + Dl] = v
    free_seeds = {s for s in range(n) if seed_map[s][0] in free}
    return seedlabs, (out, free_seeds), "ok"


def generate_u1(short_id, K, *, margin=4, verbose=False):
    """BPS-free elementary-trace record for a u1 entry (the u1 path of
    `elem_traces.generate`).  Tr(1) via one BPS call; the seeds via the
    orbit-reduced forward bootstrap.  Raises `_BootstrapUnavailable` if a seed
    is left unpinned (e.g. missing `_rho_delta` table) so the caller can fall
    back to the per-seed BPS engine.

    The forward solve runs to `K+margin` and is truncated to `K`: the top few
    orders of a forward pass are boundary-under-constrained (the reach-aware
    pool's deepest labels only reach the frontier), so the margin pushes that
    boundary above `K` and makes the returned coefficients exact (verified
    against the frozen a5 table)."""
    if REGEN_SPECS[short_id][2] != "u1":
        raise _BootstrapUnavailable(f"{short_id}: flavour is not u1")
    mod, pfx = _load_standalone(short_id)
    rank = len(getattr(mod, f"{pfx}_MULT_GENS_LATTICE")[0])
    Ki = K + margin
    if verbose:
        print(f"[{short_id}] u1 bootstrap: Tr(1) via Nahm-sum (spec) at K={Ki} "
              f"(spine-free) ...", flush=True)
    ident_full = _series_to_data(short_id, _vacuum_rps(short_id, Ki))
    Tr1 = frozen_qmu(ident_full)
    seedlabs, res, status = forward_bootstrap_u1(short_id, Ki, Tr1,
                                                 verbose=verbose)
    if res is None:
        raise _BootstrapUnavailable(f"{short_id}: {status}")
    Tr, free = res
    if free:
        raise _BootstrapUnavailable(
            f"{short_id}: {len(free)} seeds unpinned (check {pfx}_RHO_DELTA)")
    idxs = [sl[0][0] for sl in seedlabs]
    orbits = {}
    for j in range(len(idxs)):
        d = {q: {(mu,): c for mu, c in mud.items() if c}
             for q, mud in Tr[j].items() if q <= K and any(mud.values())}
        orbits[idxs[j]] = {q: t for q, t in d.items() if t}
    if verbose:
        print(f"[{short_id}] u1 bootstrap pinned all {len(idxs)} seeds "
              f"(BPS-free, over-determined + consistent)", flush=True)
    return {
        "K": K,
        "flavor": "u1",
        "fold": fold_policy(short_id),
        "identity": {q: t for q, t in ident_full.items() if q <= K},
        "orbits": orbits,
    }
