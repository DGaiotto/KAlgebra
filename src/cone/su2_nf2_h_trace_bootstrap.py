"""Spine-free magnetic-trace bootstrap for `A_𝖖[SU(2)+N_f=2]` over Spin(4).

The magnetic-sector trace is solved in the **Spin(4) = SU(2)_L × SU(2)_R
character basis** (so every anchor is a class function by construction —
Weyl-invariant), by a global linear solve combining

  * ρ²-twisted **cyclicity** `Tr(xy)=Tr(ρ²(y)x)` (all output q-orders), and
  * **orthonormality** `I_{a,b}=Tr(ρ(L_a)·L_b)=δ_{a,b}+O(q)` (output q≤0),

with Clebsch–Gordan fusion done by χ-expansion.  Seeds: the closed-form
Wilson tower `tr_W` (Schur F).  Inputs: the validated spine-free `multiply`
(`su2_nf2_h_multiply.multiply_native`) and `ρ(H_n)=H_{n-2}` (⇒ ρ²(H_n)=H_{n-4},
the 4−N_f shift).  No BPS/RG/quantum-torus engine on any path.

Anchors `V_{m,e}=Tr(L_{(m,e)})` (m H-letters, electric sum e) fold by the
gauge period-2m reflective rule `e_anc=min(e mod 2m, 2m−e mod 2m)` (the
Nf-independent SU(2) cone structure).  Cartan coefficients are `RLaurent`
over `AbelianZPlusRing(rank=2)` (μ_L, μ_R); converted to Spin(4) characters
at the boundary.  Validated anchor-for-anchor and via orthonormality against
a BPS-quiver oracle (a derivation not included in this repository).
"""
from __future__ import annotations
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from zplus_ring import AbelianZPlusRing, RElement, RLaurent
from su2_nf2_h_multiply import multiply_native
from su2_nf2_h_trace import tr_W, _cartan_to_su2     # tr_W: Cartan RLaurent

_R = AbelianZPlusRing(rank=2)
RHO2 = 4          # ρ²(H_n) = H_{n-4}  (shift 4 − N_f = 2 per ρ)


def _z():
    return RLaurent(_R, {})


def _one():
    return RLaurent(_R, {0: _R.one()})


def _trunc(f, K):
    return RLaurent(_R, {e: c for e, c in f.coeffs.items() if e <= K})


# --- gauge-label helpers: native (h_factors,(0,0)) <-> (m,e) ---------------
def _native(m, e):
    if m == 0:
        return ((), (0, 0)) if e == 0 else (((('W', e), 1),), (0, 0))
    base, rem = divmod(e, m)
    ent = []
    if m - rem > 0:
        ent.append((base, m - rem))
    if rem > 0:
        ent.append((base + 1, rem))
    return (tuple(ent), (0, 0))


def _seed_of(hf):
    """gauge native h_factors -> (m,e); m=0 = Wilson(e=power)/identity(e=0)."""
    if not hf:
        return (0, 0)
    first = hf[0]
    if isinstance(first[0], tuple) and first[0][0] == 'W':
        return (0, first[0][1])
    m = sum(x for _, x in hf)
    e = sum(n * x for n, x in hf)
    return (m, e)


def _cartan_mul(seedA, seedB):
    """(m,e)-gauge × (m,e)-gauge -> {seed: RLaurent} folding flavour into μ."""
    a = _native(*seedA)
    b = _native(*seedB)
    out = {}
    for lab, co in multiply_native(a, b).terms.items():
        hf, (wL, wR) = lab
        s = _seed_of(hf)
        rl = RLaurent(_R, {e: _R.basis_element((wL, wR)) * v
                           for e, v in co._coeffs.items()})
        out[s] = out.get(s, _z()) + rl
    return {k: v for k, v in out.items() if not v.is_zero()}


def _multi_letter(indices):
    """H_{a0}·…·H_{a_{n-1}} -> {seed: RLaurent}."""
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
    """seed=(m,e); m=0 = Wilson. Returns (anchor_or_None, mu_factor, const)."""
    m, e = seed
    if m == 0:
        return (None, _one(), _trunc(tr_W(e, q_max=K), K)) if e >= 0 else (None, _z(), _z())
    period = 2 * m
    r = e % period
    return ((m, min(r, period - r)), _one(), _z())


# --- cyclicity and orthonormality equations --------------------------------
def _cyc_eq(indices, K):
    """Tr(H_{i0}…) − Tr(H_{last−4}·H_{i0}…H_{last−1}) = 0 -> ({anchor:coef}, const)."""
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


def _orth_eq(a, b, K):
    """I_{Ha,Hb}=Tr(ρ(Ha)·Hb)=δ+O(q); ρ(H_a)=H_{a−2}.  Returns ({anchor:coef},
    const) with Σ coef·T_anchor = const, imposed at output q≤0."""
    prod = _multi_letter([a - 2, b])
    eq = {}
    const = _z()
    for s, co in prod.items():
        anc, mu, c = _tr_of_seed(s, K)
        if anc is None:
            const = const - _trunc(co * c, K)
        else:
            eq[anc] = eq.get(anc, _z()) + co * mu
    if a == b:
        const = const + RLaurent(_R, {0: RElement(_R, {(0, 0): 1})})
    return eq, const


def _equations(maxm, buf):
    """Well-conditioned cyclicity equations spanning anchors up to charge maxm."""
    eqs = []
    for p in [(-2, -1), (-2, 1), (-2, 2), (-1, 0), (-1, 1), (-1, 2),
              (0, 1), (1, 2), (-1, 3), (0, 2)]:
        eqs.append(_cyc_eq(p, buf))
    for m in range(2, maxm + 1):
        for k in range(m + 1):
            eqs.append(_cyc_eq(tuple([0] * (m - k) + [1] * k), buf))   # H_0^{m-k} H_1^k
    return eqs


# --- Spin(4) irrep helpers -------------------------------------------------
def _wts_of_irrep(nL, nR):
    return [(wL, wR) for wL in range(-nL, nL + 1, 2) for wR in range(-nR, nR + 1, 2)]


def _irreps_upto(k):
    return [(nL, nR) for nL in range(0, k + 1) for nR in range(0, k + 1)]


def _chi_relt(nL, nR):
    return RElement(_R, {w: 1 for w in _wts_of_irrep(nL, nR)})


# --- the solve: global cyclicity + orthonormality in the irrep basis -------
_FULL_CACHE = {}
def solve_anchors_full(K, maxm=2, idx=5):
    """Solve the magnetic anchors `V_{m,e_anc}` as Spin(4)-irrep q-series.

    Combines cyclicity (all output q) + orthonormality (output q≤0, carrying
    the δ).  Per-equation the output order is capped at `P ≤ K + emin_eq` so a
    truncated anchor `T[q^{k>K}]` (entering via deep-negative structure
    constants) cannot corrupt a row — keeping the linear system consistent.
    Returns `{(m, e_anc): {q: {(nL,nR): int}}}`.
    """
    from fractions import Fraction as Fr
    if (K, maxm, idx) in _FULL_CACHE:
        return _FULL_CACHE[(K, maxm, idx)]
    buf = K + 12
    tagged = [(eq, c, K) for (eq, c) in _equations(maxm, buf)]      # cyclicity: all q
    for a in range(-idx, idx + 1):
        for b in range(-idx, idx + 1):
            tagged.append((*_orth_eq(a, b, buf), 0))                # orthonormality: q≤0
    anchors = sorted({a for ea, _, _ in tagged for a in ea})
    cols = [(a, k, ir) for a in anchors for k in range(1, K + 1)
            for ir in _irreps_upto(k)]
    ci = {c: i for i, c in enumerate(cols)}
    chis = {ir: _chi_relt(*ir) for ir in _irreps_upto(K)}
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
                    for ir in _irreps_upto(k):
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
            assert v.denominator == 1, (cols[c], v)
            x[cols[c]] = int(v)
    sol = {}
    for a in anchors:
        d = {k: {ir: x[(a, k, ir)] for ir in _irreps_upto(k) if x.get((a, k, ir))}
             for k in range(1, K + 1)}
        d = {k: v for k, v in d.items() if v}
        if d:
            sol[a] = d
    _FULL_CACHE[(K, maxm, idx)] = sol
    return sol


# --- public: Cartan / Spin(4) traces ---------------------------------------
def _anchor_cartan(seed, K):
    """Neutral gauge seed (m,e) -> Cartan RLaurent trace (m=0: tr_W; m≥1: solved)."""
    m, e = seed
    if m == 0:
        return _trunc(tr_W(e, q_max=K), K) if e >= 0 else _z()
    sol = solve_anchors_full(K, maxm=max(2, m))
    period = 2 * m
    r = e % period
    irr = sol.get((m, min(r, period - r)), {})
    out = {}
    for q, ird in irr.items():
        if q > K:
            continue
        wd = {}
        for (nL, nR), c in ird.items():
            for w in _wts_of_irrep(nL, nR):
                wd[w] = wd.get(w, 0) + c
        if wd:
            out[q] = RElement(_R, wd)
    return RLaurent(_R, out)


def _to_spin4_rps(cartan_rl, K):
    from zplus_ring import RPowerSeries, SU2ZPlusRing
    from tensor_zplus_ring import TensorZPlusRing
    R4 = TensorZPlusRing(SU2ZPlusRing(), SU2ZPlusRing())
    coeffs = {}
    for q, re in cartan_rl.coeffs.items():
        if 0 <= q <= K:
            wd = {w: c for w, c in re.terms.items() if c}
            if wd:
                ch = {k: v for k, v in _cartan_to_su2(wd).items() if v}
                if ch:
                    coeffs[q] = RElement(R4, ch)
    return RPowerSeries(R4, coeffs, K)


def trace_label_spin4(label, K):
    """Tr(L_label) for a flavour-NEUTRAL gauge label (h_factors,(0,0)) ->
    RPowerSeries over Spin(4)."""
    hf, (wL, wR) = label
    assert wL == 0 and wR == 0, "trace_label_spin4: neutral labels only"
    return _to_spin4_rps(_anchor_cartan(_seed_of(hf), K), K)


def inner_product_spin4(a_label, b_label, K):
    """I_{a,b}=Tr(ρ(L_a)·L_b) for neutral gauge generators -> Spin(4) RPS,
    computed as Σ_seed coef·T(seed) over the product decomposition (so the
    matter flavour content is handled correctly).  Verified δ + O(q)."""
    ha, (waL, waR) = a_label
    if ha and not isinstance(ha[0][0], tuple):
        rho_ha = (tuple(sorted((n - 2, x) for (n, x) in ha)), (-waL, -waR))
    else:
        rho_ha = (ha, (-waL, -waR))
    prod = multiply_native(rho_ha, b_label)
    acc = _z()
    for lab, co in prod.terms.items():
        hf, (wL, wR) = lab
        coef = RLaurent(_R, {e: _R.basis_element((wL, wR)) * v
                             for e, v in co._coeffs.items()})
        acc = acc + coef * _anchor_cartan(_seed_of(hf), K)
    return _to_spin4_rps(_trunc(acc, K), K)


__all__ = ["solve_anchors_full", "trace_label_spin4", "inner_product_spin4"]
