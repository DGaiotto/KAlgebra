"""Forward-triangular orthonormality trace bootstrap for `U1E7ConeKAlgebra`
(u(1)-gauged E7) — the "orthonormality fixes traces up to Tr(1)" demo for this
QTCone cone class.

Mirrors `u1aodd_trace_bootstrap` (the proven forward sweep, order k=1..K solving
the small per-order frontier).  The trace of a single cone-word `L_c` is taken
**flat** — the `ρ²`-canonical word itself is the trace symbol (no peeling) — so
it is trivially peel-order independent.  The cyclicity that relates distinct
words is supplied explicitly as the **one-step cyclicity relations**

    Tr(L_w) = q^{-e}·Σ_d C^d · Tr(ρ²-canon d),   L_{g1}·L_rest = q^e·L_w,
                                                  d ∈ L_rest·L_{ρ⁻²g1}

(one per clean atom-peel `g1` of `w`; valid identities by `ρ²`-twisted
cyclicity `Tr(L_a L_b)=Tr(ρ²(L_b) L_a)`, i.e. `Tr(L_w)=Tr(L_rest·L_{ρ⁻²g1})`).
Feeding these as exact homogeneous (`=0`) constraints into the *same* forward
sweep lets the per-order Fraction solver fix cyclicity + orthonormality +
deep-powers together — and being one-step they are valid (no recursion, so none
of the peel-order-dependent "premature seed" inconsistency of a recursive
reduction).

Constraints (all enforced at `t ≤ 0`, where they are triangular — highest
unknown order at the extraction order):
  * **deep powers** `Tr(L^p) = O(q)` — no `q^{≤0}` part.
  * **self-norms / cross-orthonormality** `I_{a,b}=Tr(ρ(a)·b)=δ_{ab}+O(q)`.
  * **one-step cyclicity relations** `Tr(L_w)-q^{-e}Σ_d C^d Tr(d) = 0`.
Magnetic (`c0≠0`) seeds are 0; the identity is `Tr(1)` (lazy vacuum recipe) and
the gauge v-tower `E^n` is supplied the same way (the gauge/vacuum sector
orthonormality alone does not reach); every other `c0=0` word is an unknown the
sweep fixes.

The forward sweep is run to `K+margin` and truncated to `K`; certified — raises
if a constraint is contradictory, under-determined at order ≤ K, or a returned
coefficient is non-integer.
"""
from __future__ import annotations

from fractions import Fraction

from laurent_poly import LaurentPoly
from u1e7_cyclic_reduce import _nz, _lp, _natoms


def _qint_series(rps):
    out = {}
    for q, v in rps.coeffs.items():
        iv = sum(int(x) for x in v.terms.values()) if hasattr(v, "terms") else int(v)
        if iv:
            out[int(q)] = iv
    return out


def rho2_rep(A, label):
    return A._canonical_rho2_orbit_rep(label)


def _pow(A, atom, a):
    return A.cone_data().from_cone_label(frozenset({atom}), {atom: a})


def flat_red(A, elt):
    """`{word_rep: LaurentPoly}` = `Tr(elt)` in `ρ²`-canonical word symbols:
    canonicalise each `c0=0` cone-term (NO peeling ⇒ peel-order independent).
    Magnetic (`c0≠0`) terms trace to 0 and are dropped."""
    out: dict = {}
    for c, co in _nz(elt).items():
        rep = A._canonical_rho2_orbit_rep(c)
        if A._mag_charge(rep) != 0:
            continue
        cur = out.get(rep)
        v = _lp(co)
        out[rep] = v if cur is None else cur + v
    return {s: p for s, p in out.items() if not p.is_zero()}


def peel_relations(A, w):
    """One-step cyclicity relations for word `w` (`nlet≥2`): a list of
    `{word_rep: LaurentPoly}` each of which traces to `0` identically,

        Tr(L_w) - q^{-e}·Σ_d C^d·Tr(ρ²-canon d) = 0,

    one per clean atom-peel `g1` (`L_{g1}·L_rest = q^e·L_w`, single-term)."""
    rels = []
    nlet, gens, powers = _natoms(A, w)
    if nlet <= 1:
        return rels
    cd = A.cone_data()
    for g1 in [g for g in gens if powers[g] > 0]:
        rp = dict(powers)
        rp[g1] -= 1
        rgens = frozenset(g for g in gens if rp.get(g, 0) > 0)
        rest = cd.from_cone_label(rgens, {g: rp[g] for g in rgens})
        m1 = _nz(A.multiply(g1, rest))
        if w not in m1 or len(m1) != 1:          # not a clean single-term peel
            continue
        e_lp = _lp(m1[w])
        (e_exp,), = [tuple(e_lp._coeffs)]
        inv = LaurentPoly({-e_exp: 1})
        g1i = A.rho_inverse(A.rho_inverse(g1))   # ρ⁻²(g1)
        daughters = _nz(A.multiply(rest, g1i))
        rel = {w: LaurentPoly({0: 1})}
        for d, Cd in daughters.items():
            drep = A._canonical_rho2_orbit_rep(d)
            if A._mag_charge(drep) != 0:
                continue
            cur = rel.get(drep)
            v = -(inv * _lp(Cd))
            rel[drep] = v if cur is None else cur + v
        rel = {s: p for s, p in rel.items() if not p.is_zero()}
        if rel:
            rels.append(rel)
    return rels


def solve_chord_seeds(A, K, *, margin=3, pcap=10, verbose=False):
    """`{word_rep: {q:int}}` for every `c0=0` word fixed by the forward-triangular
    orthonormality sweep through order `K`.  Raises if a constraint is
    contradictory, under-determined at order ≤ K, or non-integer (certified exact
    through `K`)."""
    Ks = K + margin

    _kc: dict = {}

    def known(s):
        """`{q:int}` for a known seed: magnetic (`c0≠0`) → 0; the gauge v-tower
        `E^n` (`chord=()`) → the lazy vacuum recipe.  Every other `c0=0` word →
        None (orthonormality + cyclicity fix it, up to this gauge/vacuum sector).
        """
        chord, (c0, c1) = s
        if c0 != 0:
            return {}
        if chord == ():
            if c1 not in _kc:
                _kc[c1] = _qint_series(A._v_tower_trace(c1, Ks + 4))
            return _kc[c1]
        return None

    pool = []   # (unk {rep:{q:int}}, kser {q:int}, delta:bool)

    def add(reduction, delta):
        unk, kser = {}, {}
        for s, poly in reduction.items():
            cs = {int(e): int(c) for e, c in poly._coeffs.items() if c}
            if not cs:
                continue
            kn = known(s)
            if kn is not None:
                for q, rc in cs.items():
                    for qi, vi in kn.items():
                        kser[q + qi] = kser.get(q + qi, 0) + rc * vi
            else:
                d = unk.setdefault(s, {})
                for q, rc in cs.items():
                    d[q] = d.get(q, 0) + rc
        if unk or delta or kser:
            pool.append((unk, kser, delta))

    gens = [g for g in A.cone_data().mult_gens() if g[1][0] == 0 and g[0] != ()]
    gens += [((), (0, 1)), ((), (0, -1))]            # the gauge v-tower E^{±}
    if verbose:
        print(f"  {len(gens)} c0=0 generators", flush=True)

    words: set = set()

    def note(red):
        for s in red:
            if s[0] != () and A._mag_charge(s) == 0:
                words.add(s)

    # deep powers Tr(L^p)=O(q): grow p until the negative-q reach covers the
    # window (or pcap), capped (high-power reductions are heavy)
    for gi, g in enumerate(gens):
        for p in range(1, pcap + 1):
            try:
                red = flat_red(A, _pow(A, g, p))
            except Exception:
                break
            add(red, False)
            note(red)
            reach = -min((q for poly in red.values()
                          for q in poly._coeffs), default=0)
            if reach >= Ks:
                break
        if verbose and gi % 10 == 0:
            print(f"    deep-powers: gen {gi}/{len(gens)}", flush=True)
    # self-norms I_{a,a}=1+O(q) AND cross-orthonormality I_{a,b}=O(q) (a≠b)
    for ai, a in enumerate(gens):
        for b in gens:
            try:
                red = flat_red(A, A.multiply(A.rho(a), b))
            except Exception:
                continue
            add(red, a == b)
            note(red)
        if verbose and ai % 10 == 0:
            print(f"    ortho pairs: gen {ai}/{len(gens)}", flush=True)

    # one-step cyclicity relations (closure over the words they reach)
    seen: set = set()
    frontier = set(words)
    ncyc = 0
    while frontier:
        nf: set = set()
        for w in frontier:
            if w in seen:
                continue
            seen.add(w)
            for rel in peel_relations(A, w):
                add(rel, False)
                ncyc += 1
                for s in rel:
                    if s[0] != () and A._mag_charge(s) == 0 and s not in seen:
                        nf.add(s)
        frontier = nf
    if verbose:
        print(f"  {ncyc} cyclicity relations over {len(seen)} words", flush=True)

    # forward sweep (mirror u1aodd_trace_bootstrap); every constraint is a
    # "no q^{≤t} part" row enforced at t ≤ 0 (delta carries the I_{a,a} q⁰=1)
    Tr: dict = {}
    free_orders: set = set()
    contradictions: list = []

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
        sol, free, contra = _solve(eqs)
        if contra and ko <= K:
            contradictions.append((ko, len(contra)))
        for r, v in sol.items():
            if v != 0:
                Tr.setdefault(r, {})[ko] = v
        if ko <= K:
            free_orders |= {(r, ko) for r in free}

    if contradictions:
        raise RuntimeError(
            f"u1e7 forward bootstrap contradictory (K={K}): "
            f"{contradictions} (order, #rows)")
    if free_orders:
        ex = sorted(free_orders, key=lambda x: (x[1], repr(x[0])))[:3]
        raise RuntimeError(
            f"u1e7 forward bootstrap under-determined (K={K}): "
            f"{len(free_orders)} free, e.g. {[(e[0][0], e[1]) for e in ex]}")
    out: dict = {}
    for r, d in Tr.items():
        rd = {}
        for q, v in d.items():
            if q > K:
                continue
            if getattr(v, "denominator", 1) != 1:
                raise RuntimeError(
                    f"u1e7 forward bootstrap non-integer Tr[q^{q}]={v} (K={K})")
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
    contra = [r for r in rows if not any(r[:-1]) and r[-1] != 0]
    sol = {}
    for c, rr in piv.items():
        if not [cc for cc in range(len(cols)) if cc != c and rows[rr][cc] != 0]:
            sol[cols[c]] = rows[rr][-1]
    free = set(cols) - set(sol)
    return sol, free, contra
