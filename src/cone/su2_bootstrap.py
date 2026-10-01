"""BPS-free elementary-trace generation for the su2-flavoured finite zoo
(the [A₁,D_odd] entries a1d3 / a1d5 / a1d7), by the SU(2)-irrep orthonormality
bootstrap.

The non-abelian (SU(2)) sibling of the trivial-R `_generate_bootstrap` and the
abelian-flavour `u1_bootstrap`.  The mechanism is identical — `Tr(1)` comes from
`elem_traces._vacuum_rps` (the exact Nahm sum on the embedded BPS spectrum), the
cone-data Layer-1 reducer expresses each deep single-mult-gen
label `((i,a),)` as `Σ_s P_s(𝖖)·Tr(s)` (cheaply), and the vanishing of every
closed `𝖖^{≤0}` coefficient of `Tr(L)=δ_{L,1}+O(𝖖)` is one exact linear
equation — with one twist that distinguishes su2 from u1:

  * **the reduction coefficients are SU(2) characters that FUSE** (Clebsch–
    Gordan, `χ_a·χ_b = χ_{a+b}+…+χ_{|a−b|}`), not group-like μ-monomials.  So
    `P_s·Tr(s)` couples irrep components, and the unknowns are indexed by
    `(seed, 𝖖-order, irrep)`.  Each equation is taken **per output irrep χ_N**:
    expand the fusion exactly and collect the coefficient of `χ_N`.  This is
    still an exact *integer* linear system (the fusion structure constants are
    0/1), solved by the shared `_solve_full`.

Two simplifications make su2 the *easy* non-trivial case (the repo's flavour
theorem: ρ²-orbit folding is valid for trivial/su2, invalid for the
unit-character u1):

  * **su2 folds like trivial-R** (`fold_policy = "rho2"`): `⋆ = id` on SU(2)
    irreps (every rep is self-dual), so there is no `μ^δ` twist to carry and the
    seeds from `seed_set` are already ρ²-orbit reps — exactly the trivial-R
    structure, no orbit walk.
  * traces live **natively in the irrep basis** `{𝖖:{n:c}}` (the record format,
    via `_series_to_data`'s `_su2_decompose`), so no abelian↔irrep round-trip is
    needed: the bootstrap solves for the irrep coefficients directly.

Seed counts are small (a1d3: 2, a1d5: 4, a1d7: 6), so a single global exact
Gaussian elimination over all `(seed, 𝖖-order, irrep)` unknowns suffices — the
u1 module's ρ²-orbit reduction + forward-triangular sweep (needed for a7/e7's
65–90 seeds) is unnecessary here.

`generate_su2(short_id, K)` returns the standard elementary-trace record; it is
the su2 path of `elem_traces.generate`.  Validated BPS-free against the frozen
a1d3 (K=40) / a1d5 (K=32) / a1d7 (K=16) tables (since removed).

Known defect, OPEN and off the serving path (found 2026-09-23): `_sweep`
reads a never-solved lower-order seed value as 0.  On
a1d3 at `K = 30` a seed uncovered at orders 13–22 is read as 0 by 55 later
equations and the sweep stops "inconsistent at k=23"; no wrong value is served
(a seed left free makes the record refused).  The zoo no longer serves any
entry from this module (a1d3 from `a1d3_seeds`, a1d5 / a1d7
from closed forms); it remains a witness.
"""
from __future__ import annotations

from elem_traces import (
    _standalone_algebra, _series_to_data, _solve_full,
    _vacuum_rps, _BootstrapUnavailable, fold_policy,
)
from regen import REGEN_SPECS


# ---------------------------------------------------------------------------
# SU(2)-character (irrep) helpers
# ---------------------------------------------------------------------------

def _coeff_to_qn(co) -> dict:
    """Reduction coefficient (RLaurent over SU2ZPlusRing) -> {𝖖_exp: {irrep_n:int}}."""
    out = {}
    for q, re in co.coeffs.items():
        d = {int(n): int(c) for n, c in re.terms.items() if c}
        if d:
            out[q] = d
    return out


def _fuse(d1: dict, d2: dict) -> dict:
    """Clebsch–Gordan product of two irrep dicts {n:int}: returns {N:int} with
    `χ_a·χ_b = Σ_{N=|a−b|,step 2}^{a+b} χ_N`."""
    out: dict = {}
    for a, ca in d1.items():
        for b, cb in d2.items():
            for N in range(abs(a - b), a + b + 1, 2):
                out[N] = out.get(N, 0) + ca * cb
    return out


# ---------------------------------------------------------------------------
# pool + forward 𝖖-order sweep
# ---------------------------------------------------------------------------

def _to_pool(pool: list, reduction: dict, delta: bool, ident, pos: dict) -> None:
    """Append the reduction `Σ_s P_s·Tr(s) = δ·χ₀·[𝖖⁰]+O(𝖖)` to `pool` as
    `(P:{key:{e:{irrep:int}}}, emin, delta)` (key = "id" or seed position)."""
    P: dict = {}
    emin = 0
    for sl, co in reduction.items():
        key = "id" if sl == ident else pos.get(sl)
        if key is None:
            continue
        qn = _coeff_to_qn(co)
        if qn:
            P[key] = qn
            emin = min(emin, min(qn))
    if P:
        pool.append((P, emin, delta))


def _sweep(pool: list, Tr1: dict, K: int, *, strict: bool):
    """Forward 𝖖-order sweep.  At order `k` the frontier unknowns are
    `(seed j, irrep n')` and the equations are those whose deepest term lands on
    `𝖖^k`; each is taken per output irrep `χ_N` (Clebsch–Gordan).  Lower orders
    are already known and the identity uses `Tr(1)`.  Returns
    `(Tr:{(j,k):{n:int}}, free_seeds:set, status)`."""
    Tr: dict = {}
    free_seeds: set = set()
    for k in range(1, K + 1):
        buckets: dict = {}                     # (entry_id, N) -> [co, rhs]
        for ei, (P, emin, delta) in enumerate(pool):
            m = k + emin
            if m > 0:                          # frontier 𝖖^k not yet reached
                continue
            if delta and m == 0:
                buckets.setdefault((ei, 0), [{}, 0])[1] += 1
            for key, qn in P.items():
                for e, nud in qn.items():
                    ix = m - e
                    if key == "id":
                        if ix >= 0:            # P_id[e] ⊛ known Tr(1)[ix]
                            for N, c in _fuse(nud, Tr1.get(ix, {})).items():
                                buckets.setdefault((ei, N), [{}, 0])[1] -= c
                    elif ix == k:              # P_s[emin] ⊛ frontier χ_{n'}
                        for nprime in range(0, k + 1):
                            for N, c in _fuse(nud, {nprime: 1}).items():
                                d = buckets.setdefault((ei, N), [{}, 0])[0]
                                u = (key, nprime)
                                d[u] = d.get(u, 0) + c
                    elif 1 <= ix < k:          # P_s[e] ⊛ known Tr(s)[ix]
                        for N, c in _fuse(nud, Tr.get((key, ix), {})).items():
                            buckets.setdefault((ei, N), [{}, 0])[1] -= c
        eqs = [(co, rhs) for (co, rhs) in buckets.values() if co or rhs]
        unk = sorted({u for co, _ in eqs for u in co})
        # Reach-boundary coverage check (2026-08-31, found in the general-k
        # coefficient is carried by NO firing entry never becomes a solve
        # variable, and the sweep silently left its 𝖖^k coefficient at zero,
        # corrupting the top order(s) just below the loud inconsistency.  A
        # seed absent from every live equation is a REACH failure, not a
        # zero — mark it free so the caller raises instead of fabricating.
        all_seeds = {key for (P, _e, _d) in pool for key in P if key != "id"}
        covered = {u[0] for co, _ in eqs for u in co}
        if eqs and covered != all_seeds:
            free_seeds |= all_seeds - covered
        sol, free, consistent = _solve_full(eqs, unk)
        if not consistent:
            if strict:
                return None, None, f"inconsistent at k={k}"
            free_seeds |= {u[0] for u in unk}
            continue
        for (j, nprime), v in (sol or {}).items():
            if v != 0:
                if v.denominator != 1:
                    return None, None, f"non-integer at k={k}: {v}"
                Tr.setdefault((j, k), {})[nprime] = int(v)
        free_seeds |= {j for (j, _np) in free}
    return Tr, free_seeds, "ok"


# ---------------------------------------------------------------------------
# the bootstrap
# ---------------------------------------------------------------------------

def generate_su2(short_id: str, K: int, *, margin: int = 2,
                 verbose: bool = False) -> dict:
    """BPS-free elementary-trace record for a su2 entry (the su2 path of
    `elem_traces.generate`).  `Tr(1)` from `_vacuum_rps` (the exact Nahm sum);
    the seed traces via the SU(2)-irrep orthonormality bootstrap (forward
    𝖖-order sweep).  Raises `_BootstrapUnavailable` if the system is
    inconsistent or a seed is left unpinned; the serving path then raises
    `NotImplementedError` (there is no per-seed BPS fallback)."""
    flavor = REGEN_SPECS[short_id][2]
    if flavor != "su2":
        raise _BootstrapUnavailable(f"{short_id}: flavour {flavor!r} is not su2")
    from trace_uniqueness_proofs import seed_set, seed_reduction, _pair_poly

    A = _standalone_algebra(short_id)
    ident = A.identity()
    seedlabs = [s for s in seed_set(A) if s != ident]          # ρ²-orbit reps
    pos = {sl: p for p, sl in enumerate(seedlabs)}
    idxs = [sl[0][0] for sl in seedlabs]
    n = len(seedlabs)
    Ki = K + margin

    if verbose:
        print(f"[{short_id}] su2 bootstrap: Tr(1) via Nahm-sum (spec) at K={Ki} "
              f"(spine-free) ...", flush=True)
    Tr1 = _series_to_data(short_id, _vacuum_rps(short_id, Ki))   # {𝖖:{n:int}}

    # deep single-mult-gen identity-pairings Tr(((i,a),)) = O(𝖖)
    pool: list = []
    for idx in idxs:
        for a in range(2, 7):                       # reducer caps near degree 6
            try:
                _to_pool(pool, seed_reduction(A, ((idx, a),)), False, ident, pos)
            except Exception:
                break
    Tr, free, _ = _sweep(pool, Tr1, Ki, strict=False)
    free_seeds = {j for j in free}

    # general orthonormality pairs I_{La,Lb}=δ+O(𝖖) for the still-free seeds,
    # cheapest total degree first (first factor over the full seed set).
    deg = 2
    while free_seeds and deg <= 2 * K + 2:
        for idx in idxs:
            for a in range(1, deg):
                b = deg - a
                for jj in idxs:
                    try:
                        _to_pool(pool, _pair_poly(A, ((idx, a),), ((jj, b),)),
                                 (idx, a) == (jj, b), ident, pos)
                    except Exception:
                        pass
        Tr, free, status = _sweep(pool, Tr1, Ki, strict=True)
        if Tr is None:
            raise _BootstrapUnavailable(f"{short_id}: {status}")
        free_seeds = {j for j in free}
        if verbose:
            print(f"[{short_id}] after degree-{deg} pairs: "
                  f"{len(free_seeds)} seeds still free", flush=True)
        deg += 1

    Tr, free, status = _sweep(pool, Tr1, Ki, strict=True)
    if Tr is None:
        raise _BootstrapUnavailable(f"{short_id}: {status}")
    free_in_K = {j for j in free}

    # assemble; a seed still unpinned in [1,K] makes the record unavailable
    # (the serving path raises; there is no per-seed BPS fallback)
    if free_in_K:
        raise _BootstrapUnavailable(
            f"{short_id}: seeds {sorted(idxs[j] for j in free_in_K)} not "
            f"pinned through q^{K}")
    orbits: dict[int, dict] = {
        idx: {k: t for k in range(1, K + 1) if (t := Tr.get((j, k), {}))}
        for j, idx in enumerate(idxs)}
    if verbose:
        print(f"[{short_id}] su2 bootstrap pinned all {n} seeds "
              f"({len(pool)} pool entries, consistent)", flush=True)
    return {
        "K": K,
        "flavor": flavor,
        "fold": fold_policy(short_id),
        "identity": {e: t for e, t in Tr1.items() if e <= K},
        "orbits": orbits,
    }
