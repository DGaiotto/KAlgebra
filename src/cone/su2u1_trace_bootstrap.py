"""BPS-free su2u1 elementary-trace bootstrap — the deep-relation engine for the
[A₁,D_even] entries, on the now-ρ-clean standalones (a1d6/a1d8).

The su2 sibling (`su2_bootstrap`) handles the SU(2) fusion; the u1 sibling
(`u1_bootstrap`) handles the U(1) charge.  su2u1 is their product: the seed
traces are valued in `R(SU(2)×U(1))` (basis `(n, w₂)` = SU(2) irrep × U(1)
charge), so the forward 𝖖-sweep carries unknowns `(seed, 𝖖-order, (n, w₂))` and
each orthonormality equation is taken per **output (N, W₂)** — N from
Clebsch–Gordan fusion of the SU(2) parts, W₂ additive in the U(1) parts.

Deep relations: the **single-mult-gen power reductions** `Tr(((i,a),)) = O(𝖖)`
(`seed_reduction`, a=2..6) are honest (pure cone arithmetic, no ρ-leak) and
reach negative 𝖖 — they pin the leading seeds.  The remaining seeds are
completed by the general orthonormality pairs `I_{La,Lb}=δ+O(𝖖)`, whose ρ
carries the U(1) shift `μ^{δ}` (the `_rho_delta` table) + the ⋆ (U(1) charge
negation).  `Tr(1)` (one BPS call) is the only BPS input.

This is the "then the rest" after the ρ repair: the pairs use the bare
standalone's `rho_element`, which is an honest `⋆+μ^δ` automorphism since
`#519` (framework-level flavoured-ConeKAlgebra ρ) — so `I=Tr(ρ·)=δ+O(𝖖)` is a
true constraint and the bootstrap closes, with **no flavour-in-labels Z-form
wrapper** (`finite_su2u1_zform`, now superseded for this engine).
"""
from __future__ import annotations

from elem_traces import (
    _standalone_algebra, _bps_oracle, _solve_full, _BootstrapUnavailable,
    _vacuum_rps, fold_policy,
)
from regen import _load_standalone, REGEN_SPECS
from kalgebra import Element


# ---------------------------------------------------------------------------
# (SU(2) irrep n, U(1) charge w2) helpers
# ---------------------------------------------------------------------------

def _coeff_to_qnw(co) -> dict:
    """`RLaurent` over SU(2)×U(1) (basis `(n,m)`) → `{𝖖_exp: {(n,w2): int}}`.

    Also accepts a plain `LaurentPoly` (a flavour-trivial reduction coefficient
    — `seed_reduction` returns these for χ₀ seeds), mapping it onto the trivial
    character `(0,0)`."""
    out = {}
    coeffs = getattr(co, "coeffs", None)
    if coeffs is None:                       # LaurentPoly: trivial flavour
        for q, c in co._coeffs.items():
            if c:
                out[q] = {(0, 0): int(c)}
        return out
    for q, re in coeffs.items():
        d = {(int(n), int(m)): int(c) for (n, m), c in re.terms.items() if c}
        if d:
            out[q] = d
    return out


def _fuse2(d1: dict, d2: dict) -> dict:
    """`{(n,w2):int}` ⊛ `{(n,w2):int}` -> `{(N,W2):int}`: Clebsch–Gordan on the
    SU(2) index, additive on the U(1) charge."""
    out: dict = {}
    for (a, wa), ca in d1.items():
        for (b, wb), cb in d2.items():
            W = wa + wb
            for N in range(abs(a - b), a + b + 1, 2):
                k = (N, W)
                out[k] = out.get(k, 0) + ca * cb
    return out


def _tr1_nw_from_series(series):
    """`Tr(1)` peeled to `{𝖖:{(n,w2):int}}` (SU(2) irreps × U(1) charge) from an
    RPowerSeries over the SU(2)×U(1) ring (abelian (w1,w2) trace, top-weight-
    peeled in w1 per w2)."""
    out = {}
    for e, c in series.coeffs.items():
        terms = getattr(c, "terms", None) or {}
        per = {}
        for (w1, w2), v in terms.items():
            per.setdefault(w2, {})[w1] = per.setdefault(w2, {}).get(w1, 0) + v
        row = {}
        for w2, slc in per.items():
            rem = dict(slc)
            while rem:
                m = max(abs(x) for x in rem)
                cc = rem.get(m, rem.get(-m, 0))
                row[(m, w2)] = row.get((m, w2), 0) + cc
                for x in range(-m, m + 1, 2):
                    nv = rem.get(x, 0) - cc
                    if nv:
                        rem[x] = nv
                    else:
                        rem.pop(x, None)
        if row:
            out[e] = {k: v for k, v in row.items() if v}
    return out


def _bps_tr1_nw(B, rank, K):
    """`Tr(1)` from the BPS oracle (legacy path); peeling via
    `_tr1_nw_from_series`."""
    return _tr1_nw_from_series(B.trace((0,) * rank, K=K))


# ---------------------------------------------------------------------------
# pool + forward sweep (fusion on n, additive on w2)
# ---------------------------------------------------------------------------

def _to_pool(pool, reduction, delta, dw2, ident, pos):
    """Append `Σ_s P_s·Tr(s)=δ·[(0,0)]·[𝖖⁰]+O(𝖖)`.  `dw2` = the RHO_DELTA U(1)
    shift on the first leg (ρ on a pair; 0 for the ρ-free single-gen reductions).
    Leaky (non-seed, non-ident) reductions are skipped, not truncated."""
    P = {}
    emin = 0
    for sl, co in reduction.items():
        if sl != ident and pos.get(sl) is None:
            return False
        key = "id" if sl == ident else pos[sl]
        qnw = _coeff_to_qnw(co)
        if dw2:
            qnw = {q: {(n, w2 + dw2): c for (n, w2), c in d.items()}
                   for q, d in qnw.items()}
        if qnw:
            P[key] = qnw
            emin = min(emin, min(qnw))
    if not P:
        return False
    pool.append((P, emin, delta))
    return True


def _pair_native(A, a_word, b_word):
    """`I_{La,Lb} = Tr(ρ(L_a)·L_b)` as `{native_seed: {𝖖: {(n,w2): int}}}`,
    computed through the **framework-native honest ρ** — `#519`'s
    `ConeKAlgebra.rho_element`, `ρ(c·L_w) = ⋆(c)·μ^{δ(w)}·L_{ρ(w)}`.

    No flavour-in-labels rewrite is needed: `#519` makes the bare su2u1
    standalone's `rho_element` an honest automorphism *in place* (verified:
    0 fails on the single-gen pair sweep, a1d6/a1d8), superseding the Plan-18
    Z-form wrapper this engine used to route through.  The whole pairing then
    lives in the native SU(2)×U(1) coefficient ring, whose own multiplication
    does the Clebsch–Gordan ⊗ U(1)-charge fusion — no manual `_fuse2`.

    `ρ(L_a)·L_b = Σ_c C_c·L_c` (`C_c` an `RLaurent` over SU(2)×U(1)), and
    `Tr(L_c) = Σ_s red_{c,s}·T_s`, so the coefficient of the seed trace `T_s`
    is `Σ_c C_c·red_{c,s}` — an `RLaurent`, read off by `_coeff_to_qnw`.

    Scope: pairs **single canonical generators**.  Powered cone words
    `((i,a),)` (a≥2) are not the canonical power (product-and-peel; the
    ABSOLUTE RULE) — see the audit."""
    from trace_uniqueness_proofs import seed_reduction
    rho_a = A.rho_element(Element.basis(a_word))
    prod = A.multiply_elements(rho_a, Element.basis(b_word))
    acc: dict = {}
    for c, coeff_c in prod.terms.items():
        for seed, rl in seed_reduction(A, c).items():
            for q, nw in _coeff_to_qnw(coeff_c * rl).items():
                d = acc.setdefault(seed, {}).setdefault(q, {})
                for k, v in nw.items():
                    d[k] = d.get(k, 0) + v
    return acc


def _to_pool_qnw(pool, qnw_by_seed, delta, ident, pos):
    """Append a pre-fused pairing (`{seed: {𝖖: {(n,w2): int}}}`, the
    `_pair_native` output) to the sweep pool.  Leaky (non-seed) support →
    skip the whole entry (honest, not truncated)."""
    P: dict = {}
    emin = 0
    for sl, qnw in qnw_by_seed.items():
        qnw = {q: {k: v for k, v in d.items() if v} for q, d in qnw.items()}
        qnw = {q: d for q, d in qnw.items() if d}
        if not qnw:
            continue
        if sl != ident and pos.get(sl) is None:
            return False
        key = "id" if sl == ident else pos[sl]
        P[key] = qnw
        emin = min(emin, min(qnw))
    if not P:
        return False
    pool.append((P, emin, delta))
    return True


def _bps_seed_nw(B, gamma, K):
    """Per-seed BPS trace `Tr(L_γ)` peeled to `{𝖖: {(n,w2): int}}` — the same
    top-weight peel as `_bps_tr1_nw`, for the per-seed BPS fallback (completes
    the table when the bootstrap leaves a seed free; opt-in, since it is a BPS
    call)."""
    series = B.trace(gamma, K=K)
    out = {}
    for e, c in series.coeffs.items():
        terms = getattr(c, "terms", None) or {}
        per = {}
        for (w1, w2), v in terms.items():
            per.setdefault(w2, {})[w1] = per.setdefault(w2, {}).get(w1, 0) + v
        row = {}
        for w2, slc in per.items():
            rem = dict(slc)
            while rem:
                m = max(abs(x) for x in rem)
                cc = rem.get(m, rem.get(-m, 0))
                row[(m, w2)] = row.get((m, w2), 0) + cc
                for x in range(-m, m + 1, 2):
                    nv = rem.get(x, 0) - cc
                    if nv:
                        rem[x] = nv
                    else:
                        rem.pop(x, None)
        if row:
            out[e] = {k: v for k, v in row.items() if v}
    return out


def _sweep(pool, Tr1, K, *, strict, wmax=0):
    """Forward 𝖖-sweep; frontier unknowns `(seed j, (n,w2))`, fusion on n.
    Returns `(Tr:{(j,k):{(n,w2):int}}, free_seeds, appeared, status)`.

    `wmax` widens the U(1) candidate window (see `_cand_nw`); if 0 it is
    derived from the max |w2| appearing in `Tr1` so the frontier can absorb a
    seed's fixed flavour charge."""
    if not wmax:
        wmax = max([abs(w2) for d in Tr1.values() for (_n, w2) in d] or [0])
    Tr = {}
    free_seeds = set()
    appeared = set()
    for k in range(1, K + 1):
        buckets = {}                                   # (entry, (N,W2)) -> [co, rhs]
        for ei, (P, emin, delta) in enumerate(pool):
            m = k + emin
            if m > 0:
                continue
            if delta and m == 0:
                buckets.setdefault((ei, (0, 0)), [{}, 0])[1] += 1
            for key, qnw in P.items():
                for e, nud in qnw.items():
                    ix = m - e
                    if key == "id":
                        if ix >= 0:
                            for NW, c in _fuse2(nud, Tr1.get(ix, {})).items():
                                buckets.setdefault((ei, NW), [{}, 0])[1] -= c
                    elif ix == k:                      # frontier: ⊛ unknown χ_{(n',w2')}
                        for (nprime, w2p) in _cand_nw(k, wmax):
                            for NW, c in _fuse2(nud, {(nprime, w2p): 1}).items():
                                d = buckets.setdefault((ei, NW), [{}, 0])[0]
                                u = (key, (nprime, w2p))
                                d[u] = d.get(u, 0) + c
                    elif 1 <= ix < k:
                        for NW, c in _fuse2(nud, Tr.get((key, ix), {})).items():
                            buckets.setdefault((ei, NW), [{}, 0])[1] -= c
        eqs = [(co, rhs) for (co, rhs) in buckets.values() if co or rhs]
        unk = sorted({u for co, _ in eqs for u in co})
        appeared |= {u[0] for u in unk}
        sol, free, consistent = _solve_full(eqs, unk)
        if not consistent:
            if strict:
                return None, None, None, f"inconsistent at k={k}"
            free_seeds |= {u[0] for u in unk}
            continue
        for (j, nw), v in (sol or {}).items():
            if v != 0:
                if v.denominator != 1:
                    return None, None, None, f"non-integer at k={k}: {v}"
                Tr.setdefault((j, k), {})[nw] = int(v)
        free_seeds |= {j for (j, _nw) in free}
    return Tr, free_seeds, appeared, "ok"


def _cand_nw(k, wmax=0):
    """Candidate `(n, w2)` for a frontier unknown at 𝖖-order k: SU(2) irreps
    n ≤ k (the BPS spin grows with 𝖖-order), U(1) charge |w2| ≤ wmax + k.

    The U(1) charge is **not** 𝖖-bounded — a seed carries a fixed flavour
    charge that already shows up at its leading 𝖖-order — so the window is
    widened by the data-driven `wmax` (the max |w2| in Tr(1)).  The earlier
    `|w2| ≤ k` cap produced *false* k=1 inconsistencies: a charge-±2 seed at
    𝖖¹ (Tr(1) itself carries charge ±2) landed in a bucket no in-window
    unknown could absorb (empty-coeff / nonzero-rhs)."""
    return [(n, w2) for n in range(0, k + 1)
            for w2 in range(-(wmax + k), wmax + k + 1)]


# ---------------------------------------------------------------------------
# the bootstrap
# ---------------------------------------------------------------------------

def generate_su2u1_trace(short_id, K, *, margin=2, bps_fallback=False,
                         verbose=False, **_legacy):
    """su2u1 elementary-trace record on the **framework-native honest ρ**.

    `Tr(1)` is the only mandatory BPS call.  Seeds are pinned BPS-free from
    two honest sources: (i) the single-mult-gen power reductions
    `Tr(((i,a),)) = O(𝖖)` (pure cone arithmetic, the now-memoised reducer),
    and (ii) the orthonormality pairs `I_{La,Lb} = δ + O(𝖖)` between
    **single canonical generators**, computed through the bare standalone's
    `rho_element` — an honest `⋆+μ^δ` automorphism since `#519` (which
    superseded the Plan-18 Z-form wrapper this engine used to route through;
    `_pair_native`).

    Powered pairs (`a≥2`) are deliberately NOT used: the cone monomial
    `((i,a),)` is not the canonical power, so its pairing is not the
    orthonormality constraint and injects a genuine contradiction (k=3
    inconsistent).  Reaching the seeds the single-gen pairs leave free
    therefore needs either constructed canonical powers (product-and-peel) or,
    when `bps_fallback=True`, a per-seed BPS trace (opt-in, since it is a BPS
    call).  Default returns the BPS-free partial record (`free` lists the
    unpinned seeds)."""
    if REGEN_SPECS[short_id][2] != "su2u1":
        raise _BootstrapUnavailable(f"{short_id}: not su2u1")
    from trace_uniqueness_proofs import seed_set, seed_reduction
    mod, pfx = _load_standalone(short_id)

    A = _standalone_algebra(short_id)
    # `#519`'s framework ρ applies the honest `⋆+μ^δ` twist only when the
    # standalone carries a `_rho_delta` table (else `rho_element` is the bare
    # permutation, the broken ρ); require it.
    if not getattr(A, "_rho_delta", None):
        raise _BootstrapUnavailable(
            f"{short_id}: no _rho_delta (the U(1) shift table) — ρ would be the "
            f"bare permutation; compute it (e7 recipe) and inline it first")
    ident = A.identity()
    seedlabs = [s for s in seed_set(A) if s != ident]
    pos = {sl: p for p, sl in enumerate(seedlabs)}
    idxs = [sl[0][0] for sl in seedlabs]
    n = len(seedlabs)
    Ki = K + margin
    gens = getattr(mod, f"{pfx}_MULT_GENS_LATTICE")
    rank = len(gens[0])
    if verbose:
        print(f"[{short_id}] su2u1 trace bootstrap (native ρ): Tr(1) via "
              f"Nahm-sum (spec, spine-free) at K={Ki}", flush=True)
    Tr1 = _tr1_nw_from_series(_vacuum_rps(short_id, Ki))

    # (i) single-mult-gen power reductions (honest, ρ-free cone arithmetic;
    # some standalones reject the powered cone label — skip those, not fatal).
    # Capped at a≤4: the cone reducer is ~exponential in the power on these
    # single-gen words (a1d6 gen-0: a=4 ≈ 1 s, a=5 ≈ 10 s, a=6 ≈ 100 s — the
    # A10 memoisation does not help, the reduction tree of a pure power does not
    # re-converge), and the orthonormality pairs (shallow degree-2 products,
    # fast) already pin the full seed set without the deeper powers.
    pool = []
    for idx in idxs:
        for a in range(2, 5):
            try:
                _to_pool(pool, seed_reduction(A, ((idx, a),)), False, 0,
                         ident, pos)
            except Exception:
                break

    # (ii) single-generator orthonormality pairs via the native honest ρ (#519)
    for idx in idxs:
        for jj in idxs:
            try:
                _to_pool_qnw(pool, _pair_native(A, ((idx, 1),), ((jj, 1),)),
                             (idx == jj), ident, pos)
            except Exception:
                pass

    Tr, free, appeared, status = _sweep(pool, Tr1, Ki, strict=True)
    if Tr is None:
        raise _BootstrapUnavailable(f"{short_id}: {status}")
    free_in_K = (set(range(n)) - appeared) | free
    pinned = n - len(free_in_K)
    if verbose:
        print(f"[{short_id}] su2u1 bootstrap pinned {pinned}/{n} seeds "
              f"BPS-free ({len(pool)} pool entries)", flush=True)

    orbits = {}
    bps_filled = []
    for j, idx in enumerate(idxs):
        if j not in free_in_K:
            orbits[idx] = {k: {(nn, w2): v
                               for (nn, w2), v in Tr.get((j, k), {}).items()}
                           for k in range(1, K + 1) if Tr.get((j, k))}
        elif bps_fallback:
            nw = _bps_seed_nw(B, gens[idx], K)
            orbits[idx] = {k: d for k, d in nw.items() if 1 <= k <= K and d}
            bps_filled.append(idx)
    if bps_fallback and verbose and bps_filled:
        print(f"[{short_id}] BPS fallback filled {len(bps_filled)} seeds "
              f"-> complete table", flush=True)

    return {
        "K": K, "flavor": "su2u1", "fold": fold_policy(short_id),
        "pinned": pinned, "n_seeds": n,
        "free": [] if bps_fallback else sorted(free_in_K),
        "bps_filled": sorted(bps_filled),
        "identity": {e: t for e, t in Tr1.items() if e <= K},
        "orbits": orbits,
    }


if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "a1d6"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rec = generate_su2u1_trace(sid, K, verbose=True)
    print(f"==> {sid}: pinned {rec['pinned']}/{rec['n_seeds']} seeds BPS-free")
