"""BPS-free elementary-trace generation for the su2u1-flavoured finite zoo
(the [A₁,D_even] entries a1d4 / a1d6 / a1d8) — the **Gram/window** scaffold,
and the precise diagnosis of why it is the hard (Plan-18) case.

The coefficient ring is `R(SU(2)×U(1))` (basis `(n, k)`: SU(2) irrep n = 2·spin,
U(1) charge k).  We work in the **rank-2 abelian Cartan** `(w₁, w₂)` — both
fugacities group-like, the form the BPS oracle returns `Tr(1)` in — and recover
the SU(2)-irrep structure at the end by a Weyl top-weight peel of `w₁`
(`_peel`).  The U(1) charge `w₂` is a *unit character*: `⋆` negates it, and the
element-level `ρ(L_a)=μ₂^{δ(a)}·L_{ρ(a)}` carries a `w₂`-shift `δ` (`RHO_DELTA`,
propagated additively through the window in `_window`).

**Why su2/u1 close but su2u1 (so far) does not.**  The trivial/u1/su2 bootstraps
reduce every label to the single-mult-gen seeds via Layer 1 (`seed_reduction`),
which *bakes in* ρ²-cyclicity and yields a clean q^{≤0}-triangular system the
forward sweep solves.  For **su2u1 the Layer-1 reduction leaks**: the μ^δ
cyclicity slide drops a unit (the Plan-18 leak), so `seed_reduction` does **not**
reduce to the single-mult-gen seeds (it leaves multi-gen monomials like
`((1,2),)` irreducible) — i.e. the "seeds" are not elementary.  This module
therefore builds the orthonormality system from the *raw* structure constants
`A.multiply(A.rho(a),b)` over a **label window** (every canonical label an
unknown, `_window` BFS), so no reduction/leak — `I_{a,b}=μ₂^{δ(a)}·Σ_c C^c·T_c
=δ_{a,b}+O(𝖖)`.

**Current status — honest-fail.**  Orthonormality-only over the raw window,
solved by the seed-triangular forward `_sweep`, is **insufficient**: window
labels whose structure constants have non-negative q-reach are never pinned by
the q^{≤0} closure, and the per-order frontier sweep is not triangular on the
raw window (it reports a false inconsistency at q³ on a1d4).  Closing su2u1
needs the **full constructive `trace_uniqueness` system** — orthonormality
**plus** ρ²-cyclicity rows, solved by a **per-order GLOBAL** (sparse exact)
solve carrying the deeper canonicals as unknowns — or the **Plan-18 Z-form**
that repairs the μ^δ leak so the seed reduction (and the fast forward sweep)
becomes valid.  `generate_su2u1` raises `_BootstrapUnavailable` with this
diagnosis; the window/δ-propagation/peel machinery here is the reusable
scaffold for that follow-up.

Per-entry: a1d4 `RHO_DELTA`=0; a1d6 nonzero `RHO_DELTA`; a1d8 lacks `RHO_DELTA`
(compute via the e7 recipe, `experiments/e7_compute_rho_delta.py`) — all three
gate on the solver above.
"""
from __future__ import annotations

from elem_traces import (
    _standalone_algebra, _bps_oracle, _solve_full,
    _BootstrapUnavailable, fold_policy,
)
from regen import _load_standalone, REGEN_SPECS


# ---------------------------------------------------------------------------
# rank-2 abelian (μ₁ = SU(2) Cartan, μ₂ = U(1)) helpers
# ---------------------------------------------------------------------------

def _branch(terms: dict) -> dict:
    """SU(2)×U(1) basis dict `{(n,k):c}` -> abelian `{(w₁,k):c}` by restricting
    each SU(2) irrep χ_n to its torus weights `w₁ = -n,…,n` (step 2)."""
    out: dict = {}
    for (nn, k), c in terms.items():
        for w1 in range(-nn, nn + 1, 2):
            out[(w1, k)] = out.get((w1, k), 0) + c
    return out


def _coeff_to_q2(co) -> dict:
    """RLaurent over SU(2)×U(1) -> `{𝖖_exp: {(w₁,w₂): int}}` (abelianized)."""
    out: dict = {}
    for q, re in co.coeffs.items():
        d = {w: c for w, c in _branch(re.terms).items() if c}
        if d:
            out[q] = d
    return out


def _bps_tr1(B, rank: int, K: int) -> dict:
    """`Tr(1)` from the BPS oracle (already rank-2 abelian) as
    `{𝖖_exp: {(w₁,w₂): int}}`."""
    series = B.trace((0,) * rank, K=K)
    out: dict = {}
    for e, c in series.coeffs.items():
        terms = getattr(c, "terms", None)
        d = {tuple(w): int(v) for w, v in terms.items() if v} if terms else {}
        if d:
            out[e] = d
    return out


def _su2_peel_w1(slice_w1: dict) -> dict:
    """Decompose a Weyl-symmetric `{w₁: c}` (the SU(2)-Cartan content at fixed
    `w₂`) into SU(2) irreps `{n: c}` by top-weight peeling (raises if the
    remainder is not a non-negative-weight Weyl character)."""
    rem = {int(w): int(c) for w, c in slice_w1.items() if c}
    out: dict = {}
    while rem:
        w = max(abs(x) for x in rem)
        c = rem.get(w, rem.get(-w, 0))
        out[w] = out.get(w, 0) + c
        for x in range(-w, w + 1, 2):
            nc = rem.get(x, 0) - c
            if nc:
                rem[x] = nc
            else:
                rem.pop(x, None)
    return out


def _peel(abelian_seed: dict) -> dict:
    """`{𝖖: {(w₁,w₂): c}}` -> `{𝖖: {(n,w₂): c}}` over SU(2)×U(1): peel the `w₁`
    (SU(2)-Cartan) direction to irreps at each fixed `w₂`."""
    out: dict = {}
    for q, wd in abelian_seed.items():
        per_w2: dict = {}
        for (w1, w2), c in wd.items():
            per_w2.setdefault(w2, {})[w1] = per_w2.setdefault(w2, {}).get(w1, 0) + c
        row: dict = {}
        for w2, slc in per_w2.items():
            for nn, c in _su2_peel_w1(slc).items():
                if c:
                    row[(nn, w2)] = c
        if row:
            out[q] = row
    return out


# ---------------------------------------------------------------------------
# pool + forward 𝖖-order sweep (group-like 2-fugacity)
# ---------------------------------------------------------------------------

def _to_pool(pool: list, reduction: dict, delta: bool, dw2: int,
             ident, pos: dict) -> bool:
    """Append `Σ_s P_s·Tr(s) = δ·[(0,0)]·[𝖖⁰] + O(𝖖)` to `pool` as
    `(P, emin, w_lo, w_hi, delta)` (w_lo/w_hi = the `(w₁,w₂)` bounding box of the
    `P` weights, for the output-weight window).  `dw2` is the `RHO_DELTA`
    `w₂`-shift applied to the whole reduction (ρ on the first leg of a pair; 0
    for the ρ-free identity-pairings).

    Returns `True` if the reduction was added.  A reduction that **leaks** onto
    a non-seed canonical label (the finite-corner reducer does not always fully
    reduce su2u1 labels to the elementary seed set) is **skipped entirely**, not
    truncated — dropping a leaked term would silently corrupt the equation."""
    P: dict = {}
    emin = 0
    for sl, co in reduction.items():
        if sl != ident and pos.get(sl) is None:
            return False                         # leak onto a deeper canonical
        key = "id" if sl == ident else pos[sl]
        q2 = _coeff_to_q2(co)
        if dw2:
            q2 = {q: {(w1, w2 + dw2): c for (w1, w2), c in wd.items()}
                  for q, wd in q2.items()}
        if q2:
            P[key] = q2
            emin = min(emin, min(q2))
    if not P:
        return False
    allw = [w for q2 in P.values() for wd in q2.values() for w in wd]
    w1s = [w[0] for w in allw] or [0]
    w2s = [w[1] for w in allw] or [0]
    pool.append((P, emin, (min(w1s), min(w2s)), (max(w1s), max(w2s)), delta))
    return True


def _sweep(pool: list, Tr1: dict, K: int, *, strict: bool):
    """Forward 𝖖-order sweep; frontier unknowns `(seed j, (w₁,w₂))`, group-like
    in both fugacities.  Per pool entry and output weight `W` (over a window
    around the entry's weight box) one equation is built and the frontier
    unknown weight is back-solved as `W − (P weight)` — mirroring the u1 sweep.
    Returns `(Tr:{(j,k):{(w1,w2):int}}, free_seeds:set, status)`."""
    Tr: dict = {}
    free_seeds: set = set()
    appeared: set = set()
    for k in range(1, K + 1):
        eqs: list = []
        Wm = k + 6                                  # output-weight margin (as u1)
        for (P, emin, (w1lo, w2lo), (w1hi, w2hi), delta) in pool:
            m = k + emin
            if m > 0:
                continue
            # the seed-trace torus reach at 𝖖^k is |w| ≤ k, so outputs span the
            # P box widened by k (here Wm = k + slack).
            for W1 in range(w1lo - Wm, w1hi + Wm + 1):
                for W2 in range(w2lo - Wm, w2hi + Wm + 1):
                    co: dict = {}
                    rhs = 1 if (delta and m == 0 and (W1, W2) == (0, 0)) else 0
                    for key, q2 in P.items():
                        for e, wd in q2.items():
                            ix = m - e
                            for (a1, a2), p in wd.items():
                                u1, u2 = W1 - a1, W2 - a2
                                if key == "id":
                                    if ix >= 0:
                                        rhs -= p * Tr1.get(ix, {}).get((u1, u2), 0)
                                elif ix == k:           # frontier unknown
                                    co[(key, (u1, u2))] = \
                                        co.get((key, (u1, u2)), 0) + p
                                elif 1 <= ix < k:       # known lower order
                                    rhs -= p * Tr.get((key, ix), {}).get(
                                        (u1, u2), 0)
                    if co or rhs:
                        eqs.append((co, rhs))
        unk = sorted({u for co, _ in eqs for u in co})
        appeared |= {u[0] for u in unk}
        sol, free, consistent = _solve_full(eqs, unk)
        if not consistent:
            if strict:
                return None, None, None, f"inconsistent at k={k}"
            free_seeds |= {u[0] for u in unk}
            continue
        for (j, w), v in (sol or {}).items():
            if v != 0:
                if v.denominator != 1:
                    return None, None, None, f"non-integer at k={k}: {v}"
                Tr.setdefault((j, k), {})[w] = int(v)
        free_seeds |= {j for (j, _w) in free}
    return Tr, free_seeds, appeared, "ok"


# ---------------------------------------------------------------------------
# the bootstrap — Gram/window solve (deeper canonicals as unknowns)
# ---------------------------------------------------------------------------

def _window(A, mg: list, ident, dlt, rounds: int):
    """BFS the canonical-label window by right-multiplying by mult-gens,
    propagating the additive U(1) shift `δ` (`ρ_elt(L_a)=μ₂^{δ(a)}·L_{ρ(a)}`;
    `δ` is a unit-character grading so `δ(c)=δ(a)+δ(gen)` for every `c` in
    `L_a·L_gen`).  Returns `(labels:set, delta:{label:int})`."""
    delta = {ident: 0}
    for s in mg:
        delta[s] = dlt(s[0][0])
    frontier = list(mg)
    for _ in range(rounds):
        nf = []
        for a in frontier:
            da = delta[a]
            for s in mg:
                ds = da + delta[s]
                for c in A.multiply(a, s).terms:
                    if c not in delta:
                        delta[c] = ds
                        nf.append(c)
        frontier = nf
        if not nf:
            break
    return set(delta), delta


def generate_su2u1(short_id: str, K: int, *, margin: int = 2, rounds: int = 3,
                   verbose: bool = False) -> dict:
    """BPS-free elementary-trace record for a su2u1 entry, by the **Gram/window
    bootstrap**: every canonical label in a window is an unknown trace
    (not just the elementary seeds), so the orthonormality system
    `I_{a,b}=Tr(ρ_elt(L_a)·L_b)=δ_{a,b}+O(𝖖)` is built from the *full* structure
    constants `A.multiply(A.rho(a),b)` — no Layer-1 reduction, hence no
    seed-leak.  The element-level ρ on the first leg contributes a `μ₂^{δ(a)}`
    unit (the `RHO_DELTA` shift, propagated through the window).  `Tr(1)` is the
    only BPS call; the forward 𝖖-sweep pins the window from it, and the
    mult-gen seeds are read off.  Raises `_BootstrapUnavailable` (caller falls
    back to BPS) if `RHO_DELTA` is missing or a seed is left unreached."""
    flavor = REGEN_SPECS[short_id][2]
    if flavor != "su2u1":
        raise _BootstrapUnavailable(
            f"{short_id}: flavour {flavor!r} is not su2u1")
    from trace_uniqueness_proofs import seed_set

    mod, prefix = _load_standalone(short_id)
    rho_delta = getattr(mod, f"{prefix}_RHO_DELTA", None)
    if rho_delta is None:
        raise _BootstrapUnavailable(
            f"{short_id}: no {prefix}_RHO_DELTA (the U(1) direction map) — "
            f"compute it (e7 recipe) before the bootstrap can carry pairs")

    def dlt(i: int) -> int:
        v = rho_delta.get(i)
        return v[0] if v else 0

    A = _standalone_algebra(short_id)
    ident = A.identity()
    mg = [s for s in seed_set(A) if s != ident]
    gens = getattr(mod, f"{prefix}_MULT_GENS_LATTICE")
    rank = len(gens[0])
    B = _bps_oracle(short_id)
    Ki = K + margin
    if verbose:
        print(f"[{short_id}] su2u1 window bootstrap: Tr(1) via BPS at K={Ki} "
              f"(only BPS call) ...", flush=True)
    Tr1 = _bps_tr1(B, rank, Ki)

    # window of canonical labels (BFS) + the propagated δ-shift
    labels, delta = _window(A, mg, ident, dlt, rounds)
    cols = sorted(l for l in labels if l != ident)
    pos = {c: i for i, c in enumerate(cols)}
    if verbose:
        print(f"[{short_id}] window: {len(cols)} labels (rounds={rounds})",
              flush=True)

    # orthonormality rows I_{a,b}=μ₂^{δ(a)}·Σ_c C^c_{ρ(a),b}·Tr(L_c)=δ_{a,b}+O(𝖖),
    # a,b over the window; an equation whose product leaks outside the window is
    # skipped by `_to_pool` (never truncated).
    pool: list = []
    legs = sorted(labels)
    for a in legs:
        ra = A.rho(a)
        da = delta[a]
        for b in legs:
            _to_pool(pool, A.multiply(ra, b).terms, a == b, da, ident, pos)
    if verbose:
        print(f"[{short_id}] pool: {len(pool)} orthonormality rows", flush=True)

    Tr, free, appeared, status = _sweep(pool, Tr1, Ki, strict=True)
    if Tr is None:
        raise _BootstrapUnavailable(
            f"{short_id}: orthonormality-only window {status} — the raw window "
            f"needs the full cyclicity rows + a per-order GLOBAL solve "
            f"(constructive trace_uniqueness), not the seed-triangular forward "
            f"sweep: for su2u1 the Layer-1 μ^δ slide leaks, so the single-mult-"
            f"gen seeds are not elementary and the deeper canonicals must be "
            f"carried as unknowns with cyclicity. Plan-18 Z-form or the full "
            f"Gram solve required")

    mg_pos = {pos[s]: s[0][0] for s in mg}
    unreached = [mg_pos[j] for j in mg_pos
                 if j in free or j not in appeared]
    if unreached:
        raise _BootstrapUnavailable(
            f"{short_id}: mult-gen seeds {sorted(unreached)} unreached by the "
            f"window (rounds={rounds}; widen the window or check RHO_DELTA)")

    orbits: dict[int, dict] = {}
    for j, idx in mg_pos.items():
        orbits[idx] = _peel({k: Tr.get((j, k), {}) for k in range(1, K + 1)
                             if Tr.get((j, k))})
    if verbose:
        print(f"[{short_id}] su2u1 window bootstrap pinned {len(mg_pos)}/"
              f"{len(mg_pos)} seeds BPS-free ({len(pool)} rows, consistent)",
              flush=True)
    return {
        "K": K,
        "flavor": flavor,
        "fold": fold_policy(short_id),
        "identity": _peel({e: t for e, t in Tr1.items() if e <= K}),
        "orbits": orbits,
    }
