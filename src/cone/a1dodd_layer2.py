"""Explicit Layer-2 characters for [A₁, D_{2k+3}] = sl(2)_{-(4k+4)/(2k+3)}
(closed form, BPS-free) — the **general-k** unification of `a1d5_layer2`
(k=1, v=5) and `a1d7_layer2` (k=2, v=7).

The 2(k+1) elementary traces (the ρ²-orbit-representative single-mult-gen traces
`Tr(seed)`) plus the vacuum `Tr(1)`, as explicit affine
sl(2)_{-(4k+4)/(2k+3)} admissible-character combinations — exact to arbitrary
q-order, no bootstrap, no BPS.  Same construction as D₅/D₇, with the k+1 discrete
modules s = 1..k+1 and (u, v) = (2, 2k+3).

Seed indexing.  The 2(k+1) seeds are indexed by **(level a = 1..k+1, parity
p ∈ {0,1})**, with leading Schur term (−1)^a q^a χ_p (see
`experiments/a1dodd_general_k_scaffold.py`).  The flat `idx` (matching
`a1d5_layer2`/`a1d7_layer2`'s `seed_trace(idx)`) is

    idx ∈ [0, k]      ->  p = 0,  a = k + 1 − idx       (p=0 block, a descending)
    idx ∈ [k+1, 2k+1] ->  p = 1,  a = idx − k           (p=1 block, a ascending)

Closed-form recipe (verified against a1d5_layer2 and a1d7_layer2):

    vac          :  κ₀
    (a, p=1)     :  −Σ_{s ∈ S(a)} κ_s^anti[−a],   S(1)={1..k+1}, S(a≥2)={a−1..k}
    (a=1, p=0)   :   κ₀[−1] − κ_{k+1}^sym[−1] − (χ₁·κ_{k+1}^anti)[−1]
    (a=2, p=0)   :   κ₀[0]  + κ_1^sym[−2] − κ_{k+1}^sym[−2] − (χ₁·κ_{k+1}^anti)[−2]

The p=0 levels follow a unified closed form (dir/top κ-shifts by a parity rule
odd a → −1,−1 / even a → 0,−a, plus a middle ladder s=2..a−1 at shift −a), which
reproduces a1d5 (k=1) and a1d7 (k=2) over ALL seeds — including the former
hold-out, the diameter seed0 `(a=k+1, p=0)`.  k=1 (D₅) and k=2 (D₇) are therefore
FULLY closed-form here.

**Standing residual (k ≥ 3): the p=0, a ≥ 3 seed traces.**  The unified form is
VALIDATED only for p=0 `a ∈ {1,2}` (all k) and the diameter `a=k+1` at `k ≤ 2`;
it does NOT reproduce the BPS seed traces for p=0 with `a ≥ 3` at `k ≥ 3` (the
genuine middle levels `3 ≤ a ≤ k` AND the diameter `a=k+1 ≥ 4`) — `_recipe`
returns None there and `seed_trace` honestly raises rather than fabricate.  The
multiply side of `[A₁, D_{2k+3}]` is closed-form for ALL k (see
`a1dodd_cone_data`), so this trace residual is the one piece blocking k≥3
orthonormality; deriving the correct general-k admissible-character combination
for it is tracked in `a1dodd_cone_cracking_notes.md`.

κ machinery (σ_j numerator, verma, sym/anti modules) is identical to
`a1d5_layer2`; the verma/division helpers are shared with `a1d3_kalg`.
"""
from __future__ import annotations

from fractions import Fraction as Fr

from a1d3_kalg import (
    _bps_verma, _laurent_mul, _divide_each, _divide_by_1_minus_mu2,
    _divide_by_mu_minus_muinv, _laurent_clean, _sigma_j_add,
    _laurent_combine, _laurent_negate, _laurent_reflect_mu, _laurent_shift_mu,
)


def _uv(k: int) -> tuple[int, int]:
    """(u, v) = (2, 2k+3) for [A₁, D_{2k+3}]."""
    return 2, 2 * k + 3


# ---------------------------------------------------------------------------
# discrete-module numerator σ_j and the κ_s building blocks  (v = 2k+3)
# ---------------------------------------------------------------------------

def _sigma(K: int, s: int, u: int, v: int) -> dict:
    out: dict = {}
    jb = max(8, int((K // (2 * u * v)) ** 0.5) + 6)
    for j in range(-jb, jb + 1):
        e1 = 2 * u * v * j * j + (-2 * v + 4 * u * s) * j
        if 0 <= e1 <= K:
            _sigma_j_add(out, e1, 2 * u * j, +1)
        e2 = 2 * u * v * j * j + (-6 * v + 4 * u * s) * j + (2 * v - 2 * u * s)
        if 0 <= e2 <= K:
            _sigma_j_add(out, e2, 2 * u * j - 2, -1)
    return _laurent_clean(out)


def kappa0(K: int, k: int) -> dict:
    u, v = _uv(k)
    return _laurent_mul(_bps_verma(K),
                        _divide_each(_sigma(K, 0, u, v), _divide_by_1_minus_mu2), K)


def kappa_sym(K: int, s: int, k: int) -> dict:
    u, v = _uv(k)
    sp = _sigma(K, s, u, v)
    sm = _laurent_reflect_mu(sp)
    num = _divide_each(_laurent_combine(sp, _laurent_shift_mu(_laurent_negate(sm), 2)),
                       _divide_by_1_minus_mu2)
    return _laurent_mul(_bps_verma(K), num, K)


def kappa_anti(K: int, s: int, k: int) -> dict:
    u, v = _uv(k)
    sp = _sigma(K, s, u, v)
    sm = _laurent_reflect_mu(sp)
    num = _divide_each(_laurent_combine(sp, _laurent_negate(sm)),
                       _divide_by_mu_minus_muinv)
    return _laurent_mul(_bps_verma(K), num, K)


def _shift(d: dict, n: int) -> dict:
    return {q + n: mud for q, mud in d.items()}


def _chi1(d: dict) -> dict:
    """⊗ the SU(2) doublet χ₁ = μ + μ⁻¹."""
    return _laurent_combine(_laurent_shift_mu(d, 1), _laurent_shift_mu(d, -1))


def _scale(d: dict, c: int) -> dict:
    return {q: {m: c * v for m, v in mud.items()} for q, mud in d.items()}


# ---------------------------------------------------------------------------
# seed indexing  idx <-> (level a, parity p)
# ---------------------------------------------------------------------------

def idx_to_ap(k: int, idx: int) -> tuple[int, int]:
    """Flat seed index -> (level a, parity p)."""
    if idx < 0 or idx > 2 * k + 1:
        raise ValueError(f"idx {idx} out of range for k={k} (0..{2*k+1})")
    if idx <= k:
        return k + 1 - idx, 0          # p=0 block, a descending
    return idx - k, 1                  # p=1 block, a ascending


def ap_to_idx(k: int, a: int, p: int) -> int:
    return (k + 1 - a) if p == 0 else (k + a)


# ---------------------------------------------------------------------------
# the closed-form recipe, by (level a, parity p)
# ---------------------------------------------------------------------------
# Each recipe term: (builder, s, q-shift, χ₁-dress?, coeff).

def _recipe(k: int, a: int, p: int):
    """Closed-form κ-block recipe for seed (a, p), or None if not yet pinned.

    VALIDATED (vs a1d5 / a1d7 and the BPS D-quiver seed traces):
      * p = 1, all `a` (the clean `−Σ_s κ_s^anti[−a]` form);
      * p = 0, `a ∈ {1, 2}` (all k);
      * p = 0, `a = k+1` (the diameter) for `k ≤ 2`.

    NOT yet pinned (returns None ⇒ `seed_trace` honestly raises) — the genuinely
    open piece: **p = 0 with a ≥ 3 at k ≥ 3** (the middle p=0 levels `3 ≤ a ≤ k`
    AND the diameter `a = k+1 ≥ 4`).  The "unified parity-rule" form below does NOT
    reproduce the BPS seed traces there (e.g. at k=3 it misses the a=3 leading term
    `−𝖖³χ₀` and gives spurious negative-𝖖 content for the a=4 diameter — the BPS
    truth is the clean `{3:−χ₀, 5:−χ₂, …}` / `{4:χ₀, …}`).  Deriving the correct
    general-k admissible-character combination for these is the standing residual
    (the SU(2)-refined Schur index = affine sl(2)_{−(4k+4)/(2k+3)} characters);
    tracked in `a1dodd_cone_cracking_notes.md`."""
    if p == 1:
        S = range(1, k + 2) if a == 1 else range(a - 1, k + 1)
        return [("anti", s, -a, False, -1) for s in S]
    if k >= 3 and a >= 3:                       # genuinely-open middle/diameter p=0
        return None
    # p == 0 — unified closed form.  dir/top shifts follow a parity rule
    # (odd a → −1, −1 ; even a → 0, −a), plus the middle ladder s=2..a−1 at shift
    # −a.  Reproduces a1d5 (k=1, a=1,2) and a1d7 (k=2, a=1,2,3 incl. the diameter
    # seed0 `κ₀[−1]+κ₁ˢʸᵐ[−3]−κ₂ˢʸᵐ[−3]−(χ₁κ₂ᵃⁿᵗⁱ)[−3]−κ₃ˢʸᵐ[−1]−(χ₁κ₃ᵃⁿᵗⁱ)[−1]`),
    # i.e. p=0 a≤2 (all k) and the diameter a=k+1 at k≤2.  The a≥3 (k≥3) cases are
    # gated out above (NOT reproduced by this form — see the docstring).
    odd = (a % 2 == 1)
    shift_dir = -1 if odd else 0
    shift_top = -1 if odd else -a
    rec = [("dir", 0, shift_dir, False, 1)]
    if a >= 2:
        rec.append(("sym", 1, -a, False, 1))
    for s in range(2, a):                       # middle ladder (a ≥ 3)
        rec.append(("sym", s, -a, False, -1))
        rec.append(("anti", s, -a, True, -1))
    rec.append(("sym", k + 1, shift_top, False, -1))
    rec.append(("anti", k + 1, shift_top, True, -1))
    return rec


def _build(k: int, recipe, K: int) -> dict:
    """Assemble a recipe as a (q, μ)-Laurent to 𝖖-order K."""
    out: dict = {}
    for kind, s, sh, dress, c in recipe:
        base = (kappa0(K - sh, k) if kind == "dir"
                else kappa_sym(K - sh, s, k) if kind == "sym"
                else kappa_anti(K - sh, s, k))
        if dress:
            base = _chi1(base)
        out = _laurent_combine(out, _scale(_shift(base, sh), c))
    return {q: mud for q, mud in _laurent_clean(out).items() if q <= K}


def _fug_to_su2(mud: dict):
    """Weyl-symmetric integer-μ Laurent {μ:c} → {SU(2) irrep n: c} (or None)."""
    md: dict = {}
    for pp, c in mud.items():
        if Fr(pp).denominator != 1:
            return None
        md[int(pp)] = md.get(int(pp), 0) + c
    coeffs: dict = {}
    guard = 0
    while any(c for c in md.values()):
        guard += 1
        if guard > 4000:
            return None
        mx = max(pp for pp, c in md.items() if c)
        if mx < 0:
            return None
        c = md[mx]
        coeffs[mx] = coeffs.get(mx, 0) + c
        for kk in range(mx, -mx - 1, -2):
            md[kk] = md.get(kk, 0) - c
        md = {pp: cc for pp, cc in md.items() if cc}
    return {n: c for n, c in coeffs.items() if c}


def _to_irrep(qmu: dict, K: int) -> dict:
    out = {}
    for q, mud in qmu.items():
        if q > K:
            continue
        ch = _fug_to_su2(mud)
        if ch is None:
            raise RuntimeError(f"a1dodd_layer2: non-symmetric μ-content at 𝖖^{q}")
        if ch:
            out[q] = ch
    return out


# ---------------------------------------------------------------------------
# public API: elementary traces in SU(2)-irrep form {𝖖-power: {n: int}}
# ---------------------------------------------------------------------------

def vacuum_trace(k: int, K: int) -> dict:
    """Tr(1) = κ₀, the sl(2)_{-(4k+4)/(2k+3)} vacuum character, to 𝖖-order K."""
    return _to_irrep(_build(k, [("dir", 0, 0, False, 1)], K), K)


def vacuum_trace_pe(k: int, K: int) -> dict:
    """Tr(1) via the **plethystic-exponential closed form** of Pan–Yang
    (arXiv, "Exact non-Lagrangian Schur index in closed form", eq. 47):

        I_{D_{2k+3}(sl2,[1^2])} = PE[ (q − q^p)/((1−q)(1−q^p)) · χ_adj(z) ],
        p = 2k+3,  χ_adj = z² + 1 + z⁻²  (su(2) spin-1) .

    [A_1, D_{2k+3}] *is* their D_{2k+3}(sl2,[1^2]) (VOA su(2)_{−(4k+4)/(2k+3)}), so
    this equals the vacuum Schur index = `vacuum_trace`.  Their q = our 𝖖² (we
    grade in 𝖖, q_paper = 𝖖²), so the result is returned in 𝖖-powers (= 2·q_paper)
    to be directly comparable; output `{𝖖-power: {SU(2) hw n: int}}`.

    Unlike the σ_j/Kac–Wakimoto `vacuum_trace`, this is a trivial general-k closed
    form valid for ALL k (a useful cross-check, and the fast path past k=2).
    `K` is the 𝖖-order (so the paper q-order is K//2)."""
    from math import comb
    from collections import defaultdict
    p = 2 * k + 3
    Kp = K // 2                      # paper q-order
    # f = (q − q^p)/((1−q)(1−q^p)) · (z²+1+z⁻²)  as {(q-power, z-power): coeff}
    base = defaultdict(int)
    for i in range(Kp + 1):
        for j in range(Kp + 1):
            a = i + p * j
            if a <= Kp:
                base[a] += 1
    num = defaultdict(int)
    for a, c in base.items():
        if a + 1 <= Kp:
            num[a + 1] += c
        if a + p <= Kp:
            num[a + p] -= c
    f = defaultdict(int)
    for a, c in num.items():
        for zb in (2, 0, -2):
            f[(a, zb)] += c
    # PE: ∏_{(a,b)} (1 − q^a z^b)^{−c}; build q-series with z-Laurent coefficients.
    series = {0: {0: 1}}
    for (a, b), c in sorted(f.items()):
        if a == 0 or c == 0:
            continue
        out = defaultdict(lambda: defaultdict(int))
        for m in range(Kp // a + 1):
            coef, da, db = comb(c + m - 1, m), a * m, b * m
            for qp, zd in series.items():
                if qp + da > Kp:
                    continue
                for zp, cc in zd.items():
                    out[qp + da][zp + db] += coef * cc
        series = {qp: dict(zd) for qp, zd in out.items()}
    # decompose each q-power's z-Laurent into SU(2) irreps (top-weight peel)
    res = {}
    for qp, zd in series.items():
        md = dict(zd)
        coeffs = {}
        while any(v for v in md.values()):
            mx = max(pp for pp, v in md.items() if v)
            if mx < 0:
                break
            c = md[mx]
            coeffs[mx] = coeffs.get(mx, 0) + c
            for kk in range(mx, -mx - 1, -2):
                md[kk] = md.get(kk, 0) - c
        d = {n: c for n, c in coeffs.items() if c}
        if d:
            res[2 * qp] = d              # 𝖖-power = 2 · q_paper
    return res


def seed_trace(k: int, idx: int, K: int) -> dict:
    """Elementary trace `Tr(seed_idx)` (idx ∈ {0..2k+1}) to 𝖖-order K."""
    a, p = idx_to_ap(k, idx)
    recipe = _recipe(k, a, p)
    if recipe is None:
        raise NotImplementedError(
            f"a1dodd_layer2: p=0 a={a} (idx={idx}, k={k}) recipe not yet pinned; "
            f"only p=0 a<=2 and all p=1 are closed-form (see module docstring)")
    return _to_irrep(_build(k, recipe, K), K)


def seed_trace_ap(k: int, a: int, p: int, K: int) -> dict:
    """Elementary trace by (level a, parity p)."""
    return seed_trace(k, ap_to_idx(k, a, p), K)
