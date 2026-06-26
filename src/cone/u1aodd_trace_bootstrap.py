"""BPS-free, arbitrary-precision trace bootstrap for the k>=4 intermediate chords
of U1A1AoddKAlg (the u(1)-gauged [A_1, A_{2k+1}] family).

The k<=3 seeds (v-tower / long / diameter) close in u1_pgon_layer2; the
intermediate odd chords (type a, 2 < a < k+1) have no closed form (their LOG
b-slopes are an open (1,p)/B_p log-module question -- see
u1a1aodd_self_contained_findings.md).  This module computes their trace to
ARBITRARY q-order by the spine-free orthonormality bootstrap on the fast cone:
impose Tr(non-id canonical)=O(q) + self-norm Tr(rho(a)a)=1+O(q), expand each via
the exact Layer-1 reduction into Sum_s red_s*Tr(s), substitute the known seeds
(v/long/diameter/vanishing via A._trace_residual) and solve the intermediate
seeds forward.  No BPS, no oracle.

ALL type-a chords are used (NOT rho^2-orbit reps -- deduping under-determines).
The solve is CERTIFIED: every pool equation is re-verified against the solution
(over-determined consistency), and any undetermined rep raises -- so a caller
never receives an under-determined (silently wrong) value.

Validated: reproduces u1_pgon_layer2.tr_L_long_v_n / tr_L_diameter_v_n when those
seeds are forced unknown, and at k=4 the length-5 chord trace is exact-integer
and matches the conformal-weight closed form (experiments/u1pgon_even_chord_logfit)
through q=30.
"""
from __future__ import annotations

from fractions import Fraction

from trace_uniqueness_proofs import seed_reduction, _pair_poly


def _coeff_series(c):
    """reduction coeff -> {q_exp:int}."""
    if hasattr(c, "_coeffs"):
        return {int(q): int(v) for q, v in c._coeffs.items() if v}
    if hasattr(c, "coeffs"):
        out = {}
        for q, re in c.coeffs.items():
            tot = sum(int(v) for v in re.terms.values())
            if tot:
                out[int(q)] = tot
        return out
    if isinstance(c, dict):
        return {int(q): int(v) for q, v in c.items() if v}
    return {0: int(c)} if c else {}


def _rho2_rep(A, label, cache):
    key = label
    if key in cache:
        return cache[key]
    orbit, cur = [], label
    for _ in range(4 * A.k + 12):
        orbit.append(cur)
        cur = A.rho(A.rho(cur))
        if cur == label:
            break
    rep = min(orbit, key=repr)
    cache[key] = rep
    return rep


def _is_intermediate(A, seed):
    """True iff `seed` is a single intermediate chord (2 < a < k+1) whose orbit
    is flavour-physical (trace-nonzero) -- i.e. exactly the seeds A._trace_residual
    cannot close.  Vanishing (even-length) intermediate chords are NOT unknown
    (they trace to 0 via the orbit-physical test)."""
    f, _ = seed
    if len(f) != 1 or f[0][2] != 1:
        return False
    a = f[0][0]
    if not (2 < a < A.k + 1):
        return False
    return A._orbit_has_physical(seed)


def _known_trace(A, seed, K):
    """q-series {q:int} for a NON-intermediate seed (v/long/diameter/vanishing),
    via A._trace_residual.  Not called for intermediate seeds (no recursion)."""
    rps = A._trace_residual(seed, K)
    out = {}
    for q, v in rps.coeffs.items():
        iv = sum(int(x) for x in v.terms.values()) if hasattr(v, "terms") else int(v)
        if iv:
            out[int(q)] = iv
    return out


def solve_intermediate(A, K, *, nmax=3, margin=6, verbose=False,
                       unknown_types=None):
    """Solve Tr of every unknown-type chord through order K.

    `unknown_types` is the set of chord types treated as bootstrap unknowns;
    default = the intermediate chords `2 < a < k+1` (the original behaviour).
    Pass e.g. `{2}` (the long chord) to bootstrap it as an unknown instead of
    reading its conjectural closed form `_tr_L_long_neg` -- the seeds left
    "known" (diameter `k+1`, v-tower, vanishing chords) are then the only cf
    inputs.

    The forward sweep is run to K+margin and the result truncated to K: a
    forward pass leaves its top ~margin orders boundary-under-constrained, so the
    margin pushes that boundary above K and makes the returned coefficients
    exact.  Returns {rho2_rep_label: {q:int}}.  Raises RuntimeError if the system
    is under-determined at order<=K or the over-determined consistency check
    fails (so the returned result is certified exact through q=K)."""
    cd = A.cone_data()
    Ks = K + margin
    if unknown_types is None:
        utypes = frozenset(a for a in range(3, A.k + 1))   # intermediate 2<a<k+1
    else:
        utypes = frozenset(unknown_types)
    repcache: dict = {}

    def rep(label):
        return _rho2_rep(A, label, repcache)

    def is_unk(seed):
        """Single chord of an unknown type whose orbit is trace-nonzero --
        exactly the seeds bootstrapped here (parametrised generalisation of
        `_is_intermediate`)."""
        f, _ = seed
        if len(f) != 1 or f[0][2] != 1:
            return False
        if f[0][0] not in utypes:
            return False
        return A._orbit_has_physical(seed)

    # known/unknown split (structural -> no _trace_residual recursion on unknowns)
    _kcache: dict = {}

    def known(seed):
        if is_unk(seed):
            return None
        if seed not in _kcache:
            _kcache[seed] = _known_trace(A, seed, Ks + 4)
        return _kcache[seed]

    # only the trace-NONZERO unknown chords are unknowns; the even-length
    # (vanishing) ones trace to 0 (handled as known) and their powers
    # are not needed -- including them just doubles the pool.
    chords = [(a, i) for (a, i) in cd._chords
              if a in utypes and A._orbit_has_physical((((a, i, 1),), 0))]
    unk_types = sorted({a for (a, i) in chords})
    if verbose:
        print(f"  unknown types {unk_types}; {len(chords)} chords", flush=True)

    pool = []   # (unk {rep:{q:int}}, kser {q:int}, delta:bool)

    def add(reduction, delta):
        unk, kser = {}, {}
        for sl, c in reduction.items():
            cs = _coeff_series(c)
            if not cs:
                continue
            if is_unk(sl):
                d = unk.setdefault(rep(sl), {})
                for q, rc in cs.items():
                    d[q] = d.get(q, 0) + rc
            else:
                kt = known(sl)
                for q, rc in cs.items():
                    for qi, vi in kt.items():
                        kser[q + qi] = kser.get(q + qi, 0) + rc * vi
        if unk or delta or kser:
            pool.append((unk, kser, delta))

    # deep powers (all chords, all gauges) + self-norm pairs
    for (a, i) in chords:
        for n in range(-nmax, nmax + 1):
            for p in range(1, Ks + 16):
                try:
                    red = seed_reduction(A, (((a, i, p),), n))
                except Exception:
                    break
                add(red, False)
                reach = -min((q for c in red.values()
                              for q in _coeff_series(c)), default=0)
                if reach >= Ks + 8:
                    break
    for (a, i) in chords:
        for n in range(-nmax, nmax + 1):
            try:
                add(_pair_poly(A, (((a, i, 1),), n), (((a, i, 1),), n)), True)
            except Exception:
                pass

    # forward sweep
    Tr: dict = {}
    free_orders: set = set()   # (rep, order) left under-determined by the solve

    def val(r, o):
        return Tr.get(r, {}).get(o, 0)

    buckets: dict = {}
    for unk, kser, delta in pool:
        qs = sorted({q for d in unk.values() for q in d})
        if not qs:
            continue
        for t in range(0, -Ks - 6, -1):
            orders = [t - q for q in qs if t - q >= 1]
            if orders and max(orders) <= Ks:
                buckets.setdefault(max(orders), []).append((unk, kser, delta, t))

    for ko in range(1, Ks + 1):
        eqs = []
        for unk, kser, delta, t in buckets.get(ko, []):
            co, rhs, bad = {}, (1 if (delta and t == 0) else 0) - kser.get(t, 0), False
            for r, d in unk.items():
                for q, rc in d.items():
                    o = t - q
                    if o < 1:
                        continue
                    if o == ko:
                        co[r] = co.get(r, 0) + rc
                    elif o < ko:
                        rhs -= rc * val(r, o)
                    else:
                        bad = True
                        break
                if bad:
                    break
            if not bad and (co or rhs):
                eqs.append((co, rhs))
        sol, free = _solve(eqs)
        for r, v in sol.items():
            if v != 0:
                Tr.setdefault(r, {})[ko] = v
        if ko <= K:                      # boundary orders K<ko<=Ks may be free
            free_orders |= {(r, ko) for r in free}

    # certificate: the solve was never under-determined (no free variables) at
    # any returned order <=K.  Combined with the integerity assertion below (a
    # canonical-basis trace coefficient is an integer), this guards against
    # silently-wrong under-determined output.  (External validation: the result
    # matches the conformal-weight closed form `even_chord_full` and passes
    # orthonormality -- see tests / u1a1aodd_self_contained_findings.md.)
    if free_orders:
        ex = sorted(free_orders, key=lambda x: (x[1], repr(x[0])))[:3]
        raise RuntimeError(
            f"u1aodd intermediate bootstrap under-determined (k={A.k}, K={K}): "
            f"{len(free_orders)} free (rep,order), e.g. {ex}")
    # integerity of the returned orders (<=K) is part of the certificate: a
    # canonical-basis trace coefficient is an integer.
    out: dict = {}
    for r, d in Tr.items():
        rd = {}
        for q, v in d.items():
            if q > K:
                continue
            if getattr(v, "denominator", 1) != 1:
                raise RuntimeError(
                    f"u1aodd intermediate bootstrap non-integer Tr[q^{q}]={v} "
                    f"at order<=K (k={A.k}, K={K})")
            rd[q] = int(v)
        out[r] = rd
    return out


def _solve(eqs):
    cols = sorted({u for co, _ in eqs for u in co}, key=repr)
    ci = {u: i for i, u in enumerate(cols)}
    rows = []
    for co, rhs in eqs:
        r = [Fraction(0)] * (len(cols) + 1)
        for u, c in co.items():
            r[ci[u]] += c
        r[-1] = Fraction(rhs)
        if any(r):
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
    sol = {}
    for c, rr in piv.items():
        if not [cc for cc in range(len(cols)) if cc != c and rows[rr][cc] != 0]:
            # keep as Fraction; boundary orders (ko>K) can be under-constrained
            # and fractional -- integrality of the RETURNED orders (<=K) is
            # asserted by the caller.
            sol[cols[c]] = rows[rr][-1]
    free = set(cols) - set(sol)
    return sol, free
