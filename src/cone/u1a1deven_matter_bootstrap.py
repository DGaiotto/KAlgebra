"""BPS-free **SU(2) matter-seed** trace bootstrap for `U1A1DevenConeKAlgebra`
(`[A₁, D_{2k+2}]`, U(1)-gauged, SU(2) flavour).

The matter rays `(((i,1),), m)` have **no scalar-unit pivot** — the abelian
forward solve stalls — but the Layer-1 reduction coefficients are SU(2)
characters that **fuse** (Clebsch–Gordan), so taking each
`Tr(L)=δ+O(𝖖)` / `I_{a,b}=δ+O(𝖖)` equation **per output irrep χ_N** gives an
exact *integer* system that **does** pin them.  This is the
`su2_bootstrap` mechanism specialised to the deven cone, wrapped in the
`u1aodd_trace_bootstrap` known/unknown structure: the gauge sector (`Tr(1)` +
the v-tower `X01^m`, both signs) is **known** (supplied by the caller), the
matter rays are **solved forward** and **certified** (consistency +
integrality).  Spine-free: no BPS, no oracle.

`solve_matter(A, known_qn, K)` → `{matter ρ²-rep seed: {𝖖: {irrep: int}}}`,
covering both `m ≥ 0` and `m < 0` (the negative-X01 seeds Layer-1 emits, which
the m≥0 freeze misses).  Proven at k=1: reproduces the frozen matter seeds 8/8.

`full=True` additionally treats the **v-tower** `X01^m` as unknown and adds the
monopole sandwiches `Tr(g·s·h)=Tr(ρ²(h)·g·s)` as **all-orders** identities
(processed at every `𝖖^m`, not just the closed `𝖖^{≤0}` — their `𝖖^{>0}` content
is exactly what pins the v-tower's higher-order corrections).  This bootstraps
the **entire** trace from **`Tr(1)` alone**: validated at k=1, **100%** —
v-tower + matter match the oracle through K=4 (216/216) and K=6 (384/384).
`U1A1DevenConeKAlgebra.matter_seed_trace` uses `full=False` (matter only, v-tower
supplied); `bootstrap_all_seeds` uses `full=True` (Tr(1) the sole input).
"""
from __future__ import annotations

from fractions import Fraction

from trace_uniqueness_proofs import seed_reduction, _pair_poly
from laurent_poly import LaurentPoly
from zplus_ring import RLaurent


# --- SU(2)-irrep helpers ---------------------------------------------------

def _fuse(d1: dict, d2: dict) -> dict:
    """Clebsch–Gordan product of irrep dicts: χ_a·χ_b = Σ_{|a−b|≤N≤a+b, step2} χ_N."""
    out: dict = {}
    for a, ca in d1.items():
        for b, cb in d2.items():
            for N in range(abs(a - b), a + b + 1, 2):
                out[N] = out.get(N, 0) + ca * cb
    return out


def _coeff_to_qn(co) -> dict:
    """A reduction coefficient (RLaurent over SU(2), or a bare LaurentPoly) ->
    `{𝖖_exp: {irrep: int}}`."""
    out: dict = {}
    if isinstance(co, LaurentPoly):
        for q, v in co._coeffs.items():
            if v:
                out[int(q)] = {0: int(v)}
        return out
    for q, re in co.coeffs.items():
        d = {int(n): int(c) for n, c in re.terms.items() if c}
        if d:
            out[int(q)] = d
    return out


def rps_to_qn(rps) -> dict:
    """A trace `RPowerSeries` (SU(2) coeffs) -> `{𝖖: {irrep: int}}`."""
    out: dict = {}
    for q, re in rps.coeffs.items():
        d = {int(n): int(c) for n, c in re.terms.items() if c} \
            if hasattr(re, "terms") else ({0: int(re)} if re else {})
        if d:
            out[int(q)] = d
    return out


# --- the matter seeds and their reductions ---------------------------------

def _matter_seeds(A):
    """ρ²-rep seeds that are single mag-0 rays `(((i,1),), m)` — the matter
    sector.  Enumerated for both X01 signs to the freeze's m-range."""
    tb = A._tb
    mag0 = [i for i in range(len(tb.rays)) if tb.ray_section[i][0][1][0] == 0]
    return mag0


def _is_matter(A, seed) -> bool:
    f, _e = seed
    return len(f) == 1 and f[0][1] == 1 and \
        A._mag_charge(A._oracle_section(seed)) == 0


# --- monopole cyclicity (closes the v-tower from Tr(1)) --------------------

_U_PLUS = (((1, 1),), 0)        # X10   (ray 1, mag +1)
_U_MINUS = (((0, 1),), 0)       # X10⁻¹ (ray 0, mag −1)


def _rl(R, c):
    return c if isinstance(c, RLaurent) else RLaurent(R, dict(c._coeffs))


def _proj_to_seeds(A, elt):
    """`Tr(Element)` reduced to **fundamental** ρ²-canon seeds `{seed: RLaurent}`
    (mag-0 only): each term's label is run through `seed_reduction` — the same
    reducer the matter deep labels use — so multi-ray monomials collapse to the
    single-ray / v-tower fundamentals (not left as raw cone monomials)."""
    R = A.coefficient_ring()
    out: dict = {}
    for lab, c in elt.terms.items():
        if hasattr(c, "is_zero") and c.is_zero():
            continue
        if A._mag_charge(A._oracle_section(lab)) != 0:
            continue
        cl = _rl(R, c)
        for fseed, fc in seed_reduction(A, lab).items():
            if A._mag_charge(A._oracle_section(fseed)) != 0:
                continue
            out[fseed] = out.get(fseed, RLaurent(R)) + cl * _rl(R, fc)
    return {k: v for k, v in out.items() if not v.is_zero()}


def _sandwich(A, s, g, h):
    """`Tr(g·s·h) − Tr(ρ²(h)·g·s) = 0`, reduced to fundamental seeds (the
    ρ²-cyclicity for the monopole-dressed seed `g·s·h`)."""
    from kalgebra import Element
    R = A.coefficient_ring()

    def emul(*labs):
        acc = Element({labs[0]: LaurentPoly.one()})
        for x in labs[1:]:
            acc = A.multiply_elements(acc, Element({x: LaurentPoly.one()}))
        return acc

    def rho2(l):
        return A.rho(A.rho(l))

    lhs = _proj_to_seeds(A, emul(g, s, h))
    rhs = _proj_to_seeds(A, emul(rho2(h), g, s))
    rel: dict = {}
    for k in set(lhs) | set(rhs):
        v = lhs.get(k, RLaurent(R)) - rhs.get(k, RLaurent(R))
        if not v.is_zero():
            rel[k] = v
    return rel


def _gauge_relations(A, s):
    """The nontrivial monopole sandwiches that close the v-tower from Tr(1)
    (`u₊u₊`/`u₋u₋` are mag±2 ⇒ trivially 0).  Added as **all-orders** identities
    (`add(..., all_orders=True)`): their `𝖖^{>0}` content pins the v-tower's
    higher-order corrections, which the `𝖖^{≤0}`-only reading missed."""
    return [_sandwich(A, s, _U_PLUS, _U_MINUS),
            _sandwich(A, s, _U_MINUS, _U_PLUS)]


# --- the forward 𝖖-order sweep (per output irrep) --------------------------

def _solve_full(eqs):
    """Exact integer/Fraction Gaussian elimination.  `eqs = [(co:{unk:int},
    rhs:int)]`.  Returns `(sol:{unk:Fraction}, free:set, consistent:bool)`."""
    cols = sorted({u for co, _ in eqs for u in co}, key=repr)
    ci = {u: i for i, u in enumerate(cols)}
    rows = []
    for co, rhs in eqs:
        r = [Fraction(0)] * (len(cols) + 1)
        for u, c in co.items():
            r[ci[u]] += c
        r[-1] = Fraction(rhs)
        rows.append(r)
    ri, piv = 0, {}
    for c in range(len(cols)):
        p = next((rr for rr in range(ri, len(rows)) if rows[rr][c] != 0), None)
        if p is None:
            continue
        rows[ri], rows[p] = rows[p], rows[ri]
        pv = rows[ri][c]
        rows[ri] = [x / pv for x in rows[ri]]
        for rr in range(len(rows)):
            if rr != ri and rows[rr][c] != 0:
                f = rows[rr][c]
                rows[rr] = [a - f * b for a, b in zip(rows[rr], rows[ri])]
        piv[c] = ri
        ri += 1
    # consistency: a 0 = nonzero row
    for r in rows:
        if all(x == 0 for x in r[:-1]) and r[-1] != 0:
            return {}, set(cols), False
    sol = {}
    for c, rr in piv.items():
        if not [cc for cc in range(len(cols)) if cc != c and rows[rr][cc] != 0]:
            sol[cols[c]] = rows[rr][-1]
    free = set(cols) - set(sol)
    return sol, free, True


def _run_sweep(A, pool, seeds, pos, K, Ks):
    """The certified per-irrep forward 𝖖-sweep, factored out so the deep and
    degree-first generators share it.  Returns `(Tr, free_lowK)`; raises on
    inconsistency.  `Tr` is keyed `{(seed_index, 𝖖-order): {irrep: Fraction}}`."""
    Tr: dict = {}
    free_lowK: set = set()
    for k in range(1, Ks + 1):
        buckets: dict = {}
        for ei, (P, kser, delta, emin, all_orders) in enumerate(pool):
            m = k + emin
            if m > 0 and not all_orders:
                continue
            if delta and m == 0:
                buckets.setdefault((ei, 0), [{}, 0])[1] += 1
            for N, c in kser.get(m, {}).items():
                buckets.setdefault((ei, N), [{}, 0])[1] -= c
            for j, qn in P.items():
                for e, nud in qn.items():
                    ix = m - e
                    if ix == k:
                        for nprime in range(0, k + 1):
                            for N, c in _fuse(nud, {nprime: 1}).items():
                                d = buckets.setdefault((ei, N), [{}, 0])[0]
                                d[(j, nprime)] = d.get((j, nprime), 0) + c
                    elif 1 <= ix < k:
                        for N, c in _fuse(nud, Tr.get((j, ix), {})).items():
                            buckets.setdefault((ei, N), [{}, 0])[1] -= c
        eqs = [(co, rhs) for (co, rhs) in buckets.values() if co or rhs]
        sol, free, ok = _solve_full(eqs)
        if not ok:
            raise RuntimeError(f"deven matter bootstrap inconsistent at 𝖖^{k} "
                               f"(k={A.k}, K={K})")
        for (j, nprime), v in sol.items():
            if v != 0:
                Tr.setdefault((j, k), {})[nprime] = v
        if k <= K:
            free_lowK |= {j for (j, _n) in free}
    return Tr, free_lowK


def _emit(Tr, seeds, K):
    """`Tr[(j,k)]` sweep state -> `{seed: {𝖖:{irrep:int}}}` with the integer
    certificate (a canonical-basis trace coefficient is an integer)."""
    out: dict = {}
    for j, s in enumerate(seeds):
        d: dict = {}
        for ko in range(1, K + 1):
            rr = {}
            for n, v in Tr.get((j, ko), {}).items():
                if v == 0:
                    continue
                if getattr(v, "denominator", 1) != 1:
                    raise RuntimeError(f"deven matter bootstrap non-integer "
                                       f"Tr[{s}][𝖖^{ko}]={v}")
                rr[n] = int(v)
            if rr:
                d[ko] = rr
        out[s] = d
    return out


def solve_matter(A, known_qn, K, *, margin=4, nmax=3, full=False,
                 smart=False, verbose=False):
    """Solve every matter-ray seed trace through 𝖖-order `K`.

    `known_qn(seed) -> {𝖖:{irrep:int}}` supplies the **known** seeds; matter
    rays (and, when `full`, the v-tower) are the unknowns it must NOT return.
    With `full=True` the only known input is `Tr(1)` and the v-tower
    `X01^m` is solved too (via the monopole cyclicity), so the entire trace
    bootstraps from `Tr(1)` alone.  Returns `{seed: {𝖖:{irrep:int}}}`,
    certified (no free unknown at order ≤ K; integer coefficients).

    The default path generates the legacy deep single-ray powers `Tr(ray^p)`
    (`p` up to ≈`Ks+8`, reach-capped), computed per `(i, m, p)` directly — correct
    at all `k`.  (An X01-equivariance speedup — reduce at `m=0`, shift to all `m` —
    was tried and reverted: it is *false* for `k≥2`, where X01's cocycle with the
    rays makes the dressed reduction differ from a pure seed shift.)  The deep
    reductions blow up super-linearly in degree at `k≥2` (`seed_reduction(ray^p)`
    is 0.4s at p=4 but >100s at p=6, k=2) — the k=2 stall.

    `smart=True` (opt-in, experimental) instead generates **degree-first**
    cross-pairs (capping the reduction degree — the e8 "degree-wall" fix) with a
    degree-stability certificate; see `_solve_matter_smart`."""
    if smart and not full:
        return _solve_matter_smart(A, known_qn, K, margin=margin, nmax=nmax,
                                   verbose=verbose)
    cd = A.cone_data()
    tb = A._tb
    Ks = K + margin
    mag0 = _matter_seeds(A)

    # unknowns: each mag-0 ray × X01^m (ρ²-canon, both signs); + the v-tower
    # X01^m (m≠0) when full.
    seeds: list = []
    seen = set()
    for i in mag0:
        for m in range(-(Ks + 2), Ks + 3):
            s = tb.rho2_canon((((i, 1),), m))
            if _is_matter(A, s) and s not in seen:
                seen.add(s)
                seeds.append(s)
    if full:
        for m in range(-(Ks + 2), Ks + 3):
            if m == 0:
                continue
            s = tb.rho2_canon(((), m))
            if s not in seen and A._mag_charge(A._oracle_section(s)) == 0:
                seen.add(s)
                seeds.append(s)
    pos = {s: j for j, s in enumerate(seeds)}

    kc: dict = {}

    def known(seed):
        if seed not in kc:
            kc[seed] = known_qn(seed)
        return kc[seed]

    # pool of reductions: (unk {(pos,irrep-frontier)…}, kser {𝖖:{irrep}}, delta)
    pool: list = []

    def add(reduction, delta, all_orders=False):
        # `reduction` values are coeff objects (RLaurent/LaurentPoly) OR already
        # `{𝖖:{irrep}}` qn dicts (the X01-shift path passes the latter).
        P: dict = {}
        kser: dict = {}
        for sl, co in reduction.items():
            qn = co if isinstance(co, dict) else _coeff_to_qn(co)
            if not qn:
                continue
            if A._mag_charge(A._oracle_section(sl)) != 0:
                continue                                   # mag≠0 ⇒ Tr=0
            if sl in pos:
                P[pos[sl]] = qn
            else:
                kt = known(sl)
                for q, nud in qn.items():
                    for qi, di in kt.items():
                        f = _fuse(nud, di)
                        tgt = kser.setdefault(q + qi, {})
                        for N, c in f.items():
                            tgt[N] = tgt.get(N, 0) + c
        if P or delta or kser:
            emin = min([min(qn) for qn in P.values()] + [0]) if P else 0
            pool.append((P, kser, delta, emin, all_orders))

    # deep single-ray identity pairings (Tr=O(𝖖)) + matter self-norm pairs.
    # The deepest negative-𝖖 reach caps the power sweep (u1aodd pattern).
    #
    # NOTE (deven correctness, learned the hard way): every mag-0 ray is needed
    # and the power sweep must run until the reach is deep enough.  The ρ²-image
    # rays are NOT redundant — `Tr(ray9^p)=Tr(ray2^p)` as a *value*, but the two
    # reductions land on *different* tower seeds, so they are independent rows;
    # and capping the power omits the deep-reach rows.  Dropping either silently
    # under-constrains the solve (it stays consistent but converges to WRONG
    # values for some rays / the negative-m tower).  Keep both.
    # deep single-ray identity pairings (Tr=O(𝖖)) + matter self-norm pairs.
    # The deepest negative-𝖖 reach caps the power sweep (u1aodd pattern).
    #
    # NOTE (deven correctness, learned the hard way): every mag-0 ray is needed
    # and the power sweep must run until the reach is deep enough.  The ρ²-image
    # rays are NOT redundant — `Tr(ray9^p)=Tr(ray2^p)` as a *value*, but the two
    # reductions land on *different* tower seeds, so they are independent rows;
    # and capping the power omits the deep-reach rows.  Dropping either silently
    # under-constrains the solve (it stays consistent but converges to WRONG
    # values for some rays / the negative-m tower).  Keep both.
    #
    # (X01-equivariance was tried as a speedup — compute the m=0 reduction once
    # and shift to all m.  It is *false* for k≥2: dressing by X01^m changes the
    # reduction coefficients, not just the seed X01-charge — X01's cocycle with
    # the rays is nontrivial at k≥2 (it vanishes at k=1, where the shift was
    # accidentally exact).  So each (i, m, p) reduction is computed directly.)
    for i in mag0:
        for m in range(-nmax, nmax + 1):
            for p in range(2, Ks + 8):
                try:
                    red = seed_reduction(A, (((i, p),), m))
                except Exception:
                    break
                add(red, False)
                reach = -min((q for co in red.values()
                              for q in _coeff_to_qn(co)), default=0)
                if reach >= Ks + 4:
                    break
    for i in mag0:
        for m in range(-nmax, nmax + 1):
            try:
                add(_pair_poly(A, (((i, 1),), m), (((i, 1),), m)), True)
            except Exception:
                pass

    if full:
        # the four monopole sandwiches Tr(g·s·h)=Tr(ρ²(h)·g·s), g,h∈{u₊,u₋},
        # close the v-tower (all orders) from Tr(1).  Only the gauge sector
        # (identity + v-tower) needs them; the matter rays are already pinned by
        # their deep labels (adding their sandwiches over-constrains them).
        for s in [A.identity()] + [s for s in seeds if not s[0]]:
            for rel in _gauge_relations(A, s):
                add(rel, False, all_orders=True)

    # forward 𝖖-sweep (shared) + the no-free certificate + integer emit.
    Tr, free_lowK = _run_sweep(A, pool, seeds, pos, K, Ks)
    if free_lowK:
        raise RuntimeError(f"deven matter bootstrap under-determined "
                           f"(k={A.k}, K={K}): {len(free_lowK)} free seeds")
    return _emit(Tr, seeds, K)


def _stable(out, prev, K):
    """Degree-stability test on the **relevant** seeds (X01 dressing `|e|≤K`, the
    only ones a 𝖖^K trace reduction can hit): two successive degrees agree to 𝖖^K
    after dropping zero coefficients.  The robust certificate — a seed whose
    high-𝖖 value is not yet reached reads as 0 and changes when a deeper
    constraint arrives, so equality across a degree step means the reach is
    sufficient (the repo's two-window-stability idiom on the degree axis).
    (Seeds with `|e|>K` are out of any K-order trace's range and need not be
    pinned — `_seed_trace` honest-fails on them.)"""
    def norm(d):
        return {s: {q: v for q, v in t.items() if v}
                for s, t in d.items() if abs(s[1]) <= K and any(t.values())}
    return norm(out) == norm(prev)


def _solve_matter_smart(A, known_qn, K, *, margin=4, nmax=3, verbose=False):
    """Degree-first matter bootstrap (the e8 'degree-wall' fix).

    Same unknowns / per-irrep sweep / integer certificate as `solve_matter`, but
    the constraint pool is **cross-pair orthonormality** `I_{a,b}=δ+O(𝖖)` with
    `a = ray_i^{p1}·X01^{m1}`, `b = ray_j^{p2}·X01^{m2}`, generated by growing the
    **total cone-degree** `p1+p2` only until the result is **degree-stable**.
    The reductions stay low-degree (cheap) instead of the legacy deep single-ray
    powers (`p` up to `Ks+8`), whose Layer-1 reductions blow up with `K` — the
    k=2 stall.

    Constraints are **direct** dressed cross-pairs `I_{a,b}` (computed per
    `(i,j,p1,p2,m1,m2)`).  An earlier shift-based "leanness" (reduce at `m=0`,
    X01-shift to all `m`) was **wrong for k≥2** — `_pair_poly` is X01-equivariant
    only at k=1 (X01's cocycle with the rays vanishes there), so the shift
    produced false constraints → the 'inconsistent at 𝖖¹' bug.  Fixed by computing
    each dressed pair directly.

    ⚠ STATUS (2026-06-24): **experimental / WIP — not the default.**  The concept
    is *validated correct* at k=1 (degree-stable; matter seeds match the frozen
    oracle).  The proven takeaway: **degree-capping beats the deep-power wall**
    (`seed_reduction(ray^p)` is 0.4s at p=4 but >100s at p=6 for k=2 — the k=2
    stall).  Not yet production: the direct `m1×m2` generation over-generates
    (`rays²·m1·m2` per degree), so the **lean+selective generator** (one good
    constraint per (seed,order), single sweep) is the follow-up that makes k=2
    fast."""
    tb = A._tb
    Ks = K + margin
    mag0 = _matter_seeds(A)
    MR = K + 2                                     # X01 dressing reach (|e|≤K used)

    seeds, seen = [], set()
    for i in mag0:
        for m in range(-MR, MR + 1):
            s = tb.rho2_canon((((i, 1),), m))
            if _is_matter(A, s) and s not in seen:
                seen.add(s); seeds.append(s)
    pos = {s: j for j, s in enumerate(seeds)}

    kc: dict = {}

    def known(seed):
        if seed not in kc:
            kc[seed] = known_qn(seed)
        return kc[seed]

    pool: list = []

    def add(reduction, delta):
        P, kser = {}, {}
        for sl, co in reduction.items():
            qn = co if isinstance(co, dict) else _coeff_to_qn(co)
            if not qn:
                continue
            if A._mag_charge(A._oracle_section(sl)) != 0:
                continue
            if sl in pos:
                P[pos[sl]] = qn
            else:
                kt = known(sl)
                for q, nud in qn.items():
                    for qi, di in kt.items():
                        f = _fuse(nud, di)
                        tgt = kser.setdefault(q + qi, {})
                        for N, c in f.items():
                            tgt[N] = tgt.get(N, 0) + c
        if P or delta or kser:
            emin = min([min(qn) for qn in P.values()] + [0]) if P else 0
            pool.append((P, kser, delta, emin, False))

    # DIRECT dressed cross-pairs (no X01-shift: equivariance is k=1-only — see the
    # solve_matter docstring).  Each I_{a,b} is computed directly; reach to the
    # X01-dressed seeds comes from the first factor's dressing m1.
    prev_out = None
    for d in range(2, K + nmax + 5):
        for p1 in range(1, d):
            p2 = d - p1
            if p2 < 1 or p2 > p1:
                continue
            for i in mag0:
                for j in mag0:
                    for m1 in range(-MR, MR + 1):
                        for m2 in range(-nmax, nmax + 1):
                            a = (((i, p1),), m1)
                            b = (((j, p2),), m2)
                            if a > b:               # I_{a,b}; dedup the unordered pair
                                continue
                            try:
                                red = _pair_poly(A, a, b)
                            except Exception:
                                continue
                            add(red, a == b)
        Tr, _free = _run_sweep(A, pool, seeds, pos, K, Ks)
        out = _emit(Tr, seeds, K)
        if prev_out is not None and _stable(out, prev_out, K):
            if verbose:
                print(f"  smart matter bootstrap degree-stable at degree {d} "
                      f"(pool={len(pool)})", flush=True)
            return out
        prev_out = out
    raise RuntimeError(f"deven smart matter bootstrap not degree-stable by "
                       f"degree {K + nmax + 4} (k={A.k}, K={K})")
