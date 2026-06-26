"""Spine-free magnetic-trace bootstrap for `A_𝖖[SU(2)+N_f=3]` over SU(4).

The SU(4) lift of `su2_nf2_h_trace_bootstrap`.  The magnetic-sector trace is
solved in the **SU(4) irrep basis** (so every anchor is a class function by
construction — Weyl-invariant under S_4), by a global linear solve combining

  * ρ²-twisted **cyclicity** `Tr(xy)=Tr(ρ²(y)x)` (all output q-orders), and
  * **orthonormality** `I_{a,b}=Tr(ρ(L_a)·L_b)=δ_{a,b}+O(q)` (output q≤0),

with Clebsch–Gordan fusion done by χ-expansion (SU(4) Klimyk via
`_irrep_weights`).  Seeds: the closed-form Wilson tower `tr_W` (Schur F over
SU(4), `su2_nf3_h_trace`).  Inputs: the validated spine-free 2-letter product
`su2_nf3_h_gap_k.h_mul_h` and `ρ(H_n)=H_{n-1}` (⇒ ρ²(H_n)=H_{n-2}, the 4−N_f
shift).  No BPS/RG/quantum-torus engine on any path.

Anchors `V_{m,e}=Tr(L_{(m,e)})` fold by the gauge period-2m reflective rule
`e_anc=min(e mod 2m, 2m−e mod 2m)` (the Nf-independent SU(2) cone structure).
Cartan coefficients are `RLaurent` over `AbelianZPlusRing(rank=3)`
(μ_1, μ_2, μ_3 Dynkin); converted to SU(4) characters at the boundary via
`SU4ZPlusRing.from_abelian`.  Validated against the BPS oracle
(`su2_nf3_kalgebra`).
"""
from __future__ import annotations
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from zplus_ring import AbelianZPlusRing, RElement, RLaurent, SU4ZPlusRing
from su2_nf3_h_gap_k import h_mul_h, R_SU4, rl as _rl_su4
from su2_nf3_h_trace import tr_W                      # tr_W: Cartan RLaurent(rank=3)

_R = AbelianZPlusRing(rank=3)
_SU4 = SU4ZPlusRing()
RHO1 = 1          # ρ(H_n)   = H_{n-1}   (shift 4 − N_f = 1)
RHO2 = 2          # ρ²(H_n)  = H_{n-2}


def _z():
    return RLaurent(_R, {})


def _one():
    return RLaurent(_R, {0: _R.one()})


def _trunc(f, K):
    return RLaurent(_R, {e: c for e, c in f.coeffs.items() if e <= K})


# --- SU(4) char (in RLaurent[SU4] coeff) -> rank-3 Cartan weights -----------
def _su4_to_cartan_weights(relt) -> dict:
    """SU(4) character RElement -> {Dynkin weight (m1,m2,m3): Z mult}."""
    wd: dict = {}
    for pqr, c in relt.terms.items():
        for wt, m in _SU4._irrep_weights(pqr).items():
            wd[wt] = wd.get(wt, 0) + c * m
    return {k: v for k, v in wd.items() if v}


def _su4rl_to_cartan(rl_su4) -> RLaurent:
    """RLaurent over SU(4) chars -> RLaurent over rank-3 Cartan weights."""
    out: dict = {}
    for qp, relt in rl_su4.coeffs.items():
        wd = _su4_to_cartan_weights(relt)
        if wd:
            out[qp] = RElement(_R, wd)
    return RLaurent(_R, out)


# --- 2-letter gauge product H_a · H_b in clean seeds, char folded to μ ------
def _hh_cartan(a: int, b: int) -> dict:
    """H_a · H_b -> {clean seed (m,e): RLaurent[rank3]} (a, b any order).

    For a>b the reverse product is the BAR-conjugate of the forward (q→q⁻¹,
    flavour fixed): H_a·H_b = bar(H_b·H_a) since bar is antimultiplicative and
    fixes the canonical {H_n}.  (Oracle-verified: H_1·H_0 = q⁻¹M_1 + 1, the
    q→q⁻¹ image of H_0·H_1 = qM_1 + 1 — NOT q^{2c}·forward, which would wrongly
    put the identity at q⁻².  For N_f=2 the two coincide because gap1 = qM has
    a single q-power.)"""
    if a <= b:
        prod = h_mul_h(a, b)
    else:
        fwd = h_mul_h(b, a)
        prod = {s: RLaurent(R_SU4, {-e: r for e, r in c.coeffs.items()})
                for s, c in fwd.items() if not c.is_zero()}
    return {s: _su4rl_to_cartan(c) for s, c in prod.items()}


def _cartan_mul(seedA, seedB):
    """L_seedA · L_seedB -> {seed: RLaurent[rank3]}, ANY magnetic, via the
    now-total `_gauge_mul` word reducer (SU(4) chars folded to μ-weights)."""
    mA, eA = seedA
    mB, eB = seedB
    if mA == 1 and mB == 1:
        return _hh_cartan(eA, eB)                # validated 2-letter fast path
    from su2_nf3_h_multiply import _gauge_mul
    return {s: _su4rl_to_cartan(c) for s, c in _gauge_mul(seedA, seedB).items()}


def _multi_letter(indices):
    """Actual product H_{a0}·…·H_{a_{n-1}} -> {seed: RLaurent[rank3]}, folded
    left-to-right through `_cartan_mul` (any number of letters, exactly as the
    N_f=2 bootstrap).  Each step multiplies the running seed-map by H_n via the
    now-total spine-free multiply, so the cyclicity rows for the m-letter
    monomials H_0^{m-k}H_1^k close the magnetic-m anchor solve."""
    if not indices:
        return {(0, 0): _one()}
    cur = {(1, indices[0]): _one()}
    for n in indices[1:]:
        nxt = {}
        for s, co in cur.items():
            for s2, c2 in _cartan_mul(s, (1, n)).items():
                nxt[s2] = nxt.get(s2, _z()) + co * c2
        cur = {k: v for k, v in nxt.items() if not v.is_zero()}
    return cur


# --- seed classification: Wilson const vs magnetic anchor ------------------
def _tr_of_seed(seed, K):
    """seed=(m,e); m=0 = Wilson. Returns (anchor_or_None, mu_factor, const).

    Magnetic anchors fold by ρ²-PERIOD 2m only (`r=e mod 2m`): Tr(M_e)=Tr(M_{e-2m})
    via ρ²-invariance.  The N_f=2 reflective fold `min(r,2m-r)` is DROPPED — for
    SU(4) the reflection maps an anchor to its complex conjugate (V_{m,2m-r} =
    star V_{m,r}), not to itself (it is a true symmetry only for the self-dual
    SU(2)/Spin(4) reps of N_f≤2).  So V_{m,r}, r=0..2m-1, are kept distinct."""
    m, e = seed
    if m == 0:
        return (None, _one(), _trunc(tr_W(e, q_max=K), K)) if e >= 0 else (None, _z(), _z())
    period = 2 * m
    return ((m, e % period), _one(), _z())


# --- cyclicity and orthonormality equations --------------------------------
def _cyc_eq(indices, K):
    """Tr(H_{i0}…) − Tr(ρ²(H_last)·H_{i0}…H_{last−1}) = 0 -> ({anchor:coef}, const)."""
    lhs = _multi_letter(list(indices))
    rhs = _multi_letter([indices[-1] - RHO2] + list(indices[:-1]))

    def acc(seeds):
        A = {}
        C = _z()
        for s, co in seeds.items():
            anc, mu, const = _tr_of_seed(s, K)
            if anc is None:
                C = C + _trunc(co * const, K)
            else:
                contrib = _trunc(co * mu, K)
                if not contrib.is_zero():
                    A[anc] = A.get(anc, _z()) + contrib
        return A, C

    la, lc = acc(lhs)
    ra, rc = acc(rhs)
    eq = {}
    for k in set(la) | set(ra):
        d = la.get(k, _z()) - ra.get(k, _z())
        if not d.is_zero():
            eq[k] = d
    return eq, rc - lc


def _orth_eq_seed(seedA, seedB, K):
    """I_{L_A,L_B}=Tr(ρ(L_A)·L_B)=δ_{A,B}+O(q) for general cone-monomial seeds,
    imposed at output q≤0.  ρ(M^(m)_e)=M^(m)_{e−m}; the product L_{ρA}·L_B is the
    now-total spine-free multiply (SU(4) chars folded to μ).  These higher
    orthonormality rows (magnetic-1 × magnetic-(m−1) reaching magnetic m) pin the
    magnetic-m anchors V_{m,r}, which the generator-only orthonormality (magnetic
    ≤ 2 products) cannot reach."""
    from su2_nf3_h_multiply import _gauge_mul
    mA, eA = seedA
    rhoA = (mA, eA - mA) if mA >= 1 else seedA       # ρ shift −m (neutral basis)
    prod = _gauge_mul(rhoA, seedB)
    eq = {}
    const = _z()
    for s, co_su4 in prod.items():
        co = _su4rl_to_cartan(co_su4)
        anc, mu, c = _tr_of_seed(s, K)
        if anc is None:
            const = const - _trunc(co * c, K)
        else:
            eq[anc] = eq.get(anc, _z()) + co * mu
    if seedA == seedB:
        const = const + RLaurent(_R, {0: RElement(_R, {(0, 0, 0): 1})})
    return eq, const


def _orth_eq(a, b, K):
    """I_{Ha,Hb}=Tr(ρ(Ha)·Hb)=δ+O(q); ρ(H_a)=H_{a−1}.  Returns ({anchor:coef},
    const) with Σ coef·T_anchor = const, imposed at output q≤0.

    Tr(H_x·H_y) is taken via ρ²-cyclicity in FORWARD form: a reverse product
    (x>y) would put a negative-q coefficient on the identity (whose trace
    starts at q⁰ ⇒ uncancellable negative q); cyclicity Tr(H_xH_y)=
    Tr(ρ²(H_y)H_x)=Tr(H_{y-2}H_x) (forward, since y-2<x) moves the negative-q
    coefficients onto Wilson lines (traces O(q^{≥e})), keeping everything
    O(q^{≥0})."""
    prod = _multi_letter([a - RHO1, b])
    eq = {}
    const = _z()
    for s, co in prod.items():
        anc, mu, c = _tr_of_seed(s, K)
        if anc is None:
            const = const - _trunc(co * c, K)
        else:
            eq[anc] = eq.get(anc, _z()) + co * mu
    if a == b:
        const = const + RLaurent(_R, {0: RElement(_R, {(0, 0, 0): 1})})
    return eq, const


def _equations(maxm, buf):
    """Cyclicity equations spanning anchors up to charge maxm.

    The positive-q magnetic anchor content is pinned by cyclicity together with
    the Wilson-trace (tr_W) inhomogeneity that enters every gap≥2 product, so a
    BROAD set of pairs (all gaps 1..4 over a base range) is used to make the
    linear system full-rank."""
    eqs = []
    seen = set()
    for i0 in range(-3, 3):
        for g in range(1, 5):
            p = (i0, i0 + g)
            if p in seen:
                continue
            seen.add(p)
            eqs.append(_cyc_eq(p, buf))
    for m in range(2, maxm + 1):
        for k in range(m + 1):
            eqs.append(_cyc_eq(tuple([0] * (m - k) + [1] * k), buf))   # H_0^{m-k} H_1^k
    return eqs


# --- SU(4) irrep helpers ---------------------------------------------------
def _wts_of_irrep(p, q, r):
    return _su4_to_cartan_weights(RElement(R_SU4, {(p, q, r): 1}))


def _irreps_upto(k):
    """SU(4) irreps (p,q,r) with p+q+r ≤ k (level bound = q-order bound)."""
    return [(p, q, r) for p in range(0, k + 1) for q in range(0, k + 1 - p)
            for r in range(0, k + 1 - p - q)]


def _chi_relt(p, q, r):
    return RElement(_R, _wts_of_irrep(p, q, r))


def _nality(t):
    """SU(4) N-ality of a weight/irrep (p,q,r): (p+2q+3r) mod 4."""
    return (t[0] + 2 * t[1] + 3 * t[2]) % 4


def _anchor_nality(m, r):
    """Intrinsic N-ality of the anchor V_{m,r}=Tr(M_r): the matter-loop content
    of the m-letter cone monomial.  Each H-letter contributes the matter spinor
    N-ality (H_even ↔ 4̄ ≡ 3, H_odd ↔ 4 ≡ 1); their sum mod 4 fixes the anchor's
    SU(4) N-ality (so only irreps of this N-ality can appear in V_{m,r})."""
    if m == 0:
        return 0
    base, rem = divmod(r, m)
    letters = [base] * (m - rem) + [base + 1] * rem
    return sum(3 if n % 2 == 0 else 1 for n in letters) % 4



# --- the solve: global cyclicity + orthonormality in the irrep basis -------
_FULL_CACHE = {}
def solve_anchors_full(K, maxm=2, idx=3):
    """Solve the magnetic anchors `V_{m,e_anc}` as SU(4)-irrep q-series.

    Combines cyclicity (all output q) + orthonormality (output q≤0, carrying
    the δ).  Per-equation the output order is capped at `P ≤ K + emin_eq` so a
    truncated anchor `T[q^{k>K}]` (entering via deep-negative structure
    constants) cannot corrupt a row — keeping the linear system consistent.
    Returns `{(m, e_anc): {q: {(p,q,r): int}}}`.
    """
    from fractions import Fraction as Fr
    if (K, maxm, idx) in _FULL_CACHE:
        return _FULL_CACHE[(K, maxm, idx)]
    buf = K + 2          # output is capped at P ≤ K+emin_eq ≤ K, so structure
    #                      constants / tr_W only need accuracy to ~q^K (the old
    #                      K+12 made tr_W/char-expansion to q^16 — ~20x slower).
    tagged = [(eq, c, K) for (eq, c) in _equations(maxm, buf)]      # cyclicity: all q
    for a in range(-idx, idx + 1):
        for b in range(-idx, idx + 1):
            tagged.append((*_orth_eq(a, b, buf), 0))                # orthonormality: q≤0
    # Higher-magnetic orthonormality (the cascade): I_{H_a, M^(m-1)_b} reaches
    # magnetic m, pinning the magnetic-m anchors V_{m,r} that the generator-only
    # rows (magnetic ≤ 2 products) leave under-determined.  m runs 3..maxm.
    for m in range(3, maxm + 1):
        period = 2 * (m - 1)
        for a in range(-idx, idx + 1):
            for b in range(period):
                tagged.append((*_orth_eq_seed((1, a), (m - 1, b), buf), 0))
                tagged.append((*_orth_eq_seed((m - 1, b), (1, a), buf), 0))
    anchors = sorted({a for ea, _, _ in tagged for a in ea})
    chis = {ir: _chi_relt(*ir) for ir in _irreps_upto(K)}
    # Each anchor V_{m,r} has a fixed intrinsic N-ality ν(m,r); only irreps of
    # that N-ality can appear in it.  Filter columns accordingly (4x fewer, and
    # — crucially — correct: a structure constant ec can carry N-ality, so the
    # full system must be solved together, NOT split into N-ality blocks).
    av = {a: _anchor_nality(*a) for a in anchors}
    irr_by_nality = {nn: [ir for ir in _irreps_upto(K) if _nality(ir) == nn]
                     for nn in range(4)}
    cols = [(a, k, ir) for a in anchors for k in range(1, K + 1)
            for ir in irr_by_nality[av[a]] if sum(ir) <= k]
    ci = {c: i for i, c in enumerate(cols)}
    Pmin = 0
    for eq, const, _ in tagged:
        for crl in eq.values():
            if crl.coeffs:
                Pmin = min(Pmin, min(crl.coeffs))
        if const.coeffs:
            Pmin = min(Pmin, min(const.coeffs))
    Pmin += 1
    rows = []
    rhsv = []
    for eq, const, pmax in tagged:
        emin_eq = 0
        for crl in eq.values():
            if crl.coeffs:
                emin_eq = min(emin_eq, min(crl.coeffs))
        pmax = min(pmax, K + emin_eq)
        for P in range(Pmin, pmax + 1):
            byu = {}
            for a, crl in eq.items():
                for k in range(1, K + 1):
                    ec = crl.coeffs.get(P - k)
                    if ec is None or ec.is_zero():
                        continue
                    for ir in irr_by_nality[av[a]]:
                        if sum(ir) > k:
                            continue
                        wd = ec * chis[ir]
                        if not wd.is_zero():
                            u = (a, k, ir)
                            byu[u] = byu.get(u, _R.zero()) + wd
            rhs = const.coeffs.get(P, _R.zero())
            weights = set(rhs.terms) | {w for wd in byu.values() for w in wd.terms}
            for w in weights:
                row = {ci[u]: Fr(wd.terms[w]) for u, wd in byu.items() if wd.terms.get(w, 0)}
                rv = Fr(rhs.terms.get(w, 0))
                if row or rv:
                    rows.append(row)
                    rhsv.append(rv)
    n = len(cols)
    M = [[Fr(0)] * n for _ in rows]
    for r, row in enumerate(rows):
        for j, v in row.items():
            M[r][j] = v
    b = list(rhsv)
    pr = 0
    piv = [-1] * n
    for c in range(n):
        sel = next((r for r in range(pr, len(M)) if M[r][c] != 0), None)
        if sel is None:
            continue
        M[pr], M[sel] = M[sel], M[pr]
        b[pr], b[sel] = b[sel], b[pr]
        pv = M[pr][c]
        M[pr] = [z / pv for z in M[pr]]
        b[pr] = b[pr] / pv
        for r in range(len(M)):
            if r != pr and M[r][c] != 0:
                f = M[r][c]
                M[r] = [z - f * y for z, y in zip(M[r], M[pr])]
                b[r] = b[r] - f * b[pr]
        piv[c] = pr
        pr += 1
    for r in range(len(M)):
        if all(z == 0 for z in M[r]) and b[r] != 0:
            raise ValueError(f"solve_anchors_full inconsistent: rhs {b[r]}")
    x = {}
    for c in range(n):
        if piv[c] >= 0 and b[piv[c]] != 0:
            v = b[piv[c]]
            # The top ~2 q-orders are under-determined (cyclicity reaches V[q^k]
            # only through P≈k+2), so a free pivot there can come out fractional;
            # keep only integer (fully pinned) coefficients.  Callers read up to
            # q^{K-2}, which are always integral.  Low-order fractions would be a
            # real inconsistency, so flag them.
            if v.denominator == 1:
                x[cols[c]] = int(v)
            elif cols[c][1] <= K - 2:
                raise AssertionError(
                    f"non-integer anchor multiplicity at pinned order: "
                    f"{cols[c]} = {v}")
    sol = {}
    for a in anchors:
        d = {k: {ir: x[(a, k, ir)] for ir in irr_by_nality[av[a]]
                 if sum(ir) <= k and x.get((a, k, ir))}
             for k in range(1, K + 1)}
        d = {k: v for k, v in d.items() if v}
        if d:
            sol[a] = d
    _FULL_CACHE[(K, maxm, idx)] = sol
    return sol


# --- public: Cartan / SU(4) traces -----------------------------------------
def _anchor_cartan(seed, K):
    """Neutral gauge seed (m,e) -> Cartan RLaurent trace (m=0: tr_W; m≥1: solved)."""
    m, e = seed
    if m == 0:
        return _trunc(tr_W(e, q_max=K), K) if e >= 0 else _z()
    # Solve to K+2 so the top output orders q^{K-1}, q^K are fully pinned (the
    # cyclicity equations constrain V[q^k] only through P≈k+2).
    sol = solve_anchors_full(K + 2, maxm=max(2, m))
    period = 2 * m
    irr = sol.get((m, e % period), {})
    out = {}
    for q, ird in irr.items():
        if q > K:
            continue
        wd = {}
        for pqr, c in ird.items():
            for w, mlt in _wts_of_irrep(*pqr).items():
                wd[w] = wd.get(w, 0) + c * mlt
        if wd:
            out[q] = RElement(_R, wd)
    return RLaurent(_R, out)


def _to_su4_rps(cartan_rl, K):
    from zplus_ring import RPowerSeries
    coeffs = {}
    for q, re in cartan_rl.coeffs.items():
        if 0 <= q <= K:
            wd = {w: c for w, c in re.terms.items() if c}
            if wd:
                try:
                    ch = _SU4.from_abelian(RElement(_R, wd))
                except ValueError:
                    continue
                if not ch.is_zero():
                    coeffs[q] = ch
    return RPowerSeries(_SU4, coeffs, K)


def trace_label_su4(label, K):
    """Tr(L_label) for a flavour-NEUTRAL gauge label (seed,(0,0,0)) ->
    RPowerSeries over SU(4)."""
    seed, w = label
    assert tuple(w) == (0, 0, 0), "trace_label_su4: neutral labels only"
    return _to_su4_rps(_anchor_cartan(seed, K), K)


def inner_product_su4(a_label, b_label, K):
    """I_{a,b}=Tr(ρ(L_a)·L_b) for neutral gauge generators (H-tower / Wilson /
    identity) -> SU(4) RPS, computed as Σ_seed coef·T(seed) over the
    generator-OPE product decomposition.  Verified δ + O(q)."""
    from su2_nf3_h_multiply import _gauge_mul
    seedA, wa = a_label
    seedB, wb = b_label
    mA, eA = seedA
    # ρ(H_a)=H_{a-1}; ρ(Wilson/identity) gauge-fixed.  (Neutral gauge
    # generators only — flavour weights are 0 on the canonical basis.)
    rhoA = (1, eA - RHO1) if mA == 1 else seedA
    prod = _gauge_mul(rhoA, seedB)
    # fold each output seed's SU(4) char coeff into Cartan weights, trace.
    acc = _z()
    for s, co in prod.items():
        acc = acc + _su4rl_to_cartan(co) * _anchor_cartan(s, K)
    return _to_su4_rps(_trunc(acc, K), K)


__all__ = ["solve_anchors_full", "trace_label_su4", "inner_product_su4"]
