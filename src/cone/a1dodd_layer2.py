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
a probe in the source repository).  The flat `idx` (matching
`a1d5_layer2`/`a1d7_layer2`'s `seed_trace(idx)`) is

    idx ∈ [0, k]      ->  p = 0,  a = k + 1 − idx       (p=0 block, a descending)
    idx ∈ [k+1, 2k+1] ->  p = 1,  a = idx − k           (p=1 block, a ascending)

Closed-form recipe (verified against a1d5_layer2, a1d7_layer2, and the
general-k bootstrap at k = 3, 4, 5):

    vac          :  κ₀
    (a, p=1)     :  −Σ_{s=⌈a/2⌉}^{⌊(v−a)/2⌋} κ_s^anti[−a]     (ALL a, k —
                    the 2026-08-31 unified WINDOW RULE, derived from
                    val(κ_s^anti) = min(4s, 2v−4s) ≥ 2a; for a ≤ 3 it
                    coincides with the older S(1)={1..k+1}, S(a≥2)={a−1..k})
    (a, p=0)     :  κ₀[−ε] + Σ_{j=1}^{⌊a/2⌋} κ_j^sym[−(2j+ε)]
                    − Σ_{t=1,3,…,2⌈a/2⌉−1} (κ_s^sym + χ₁·κ_s^anti)[−(t+1−ε)],
                    s = (v−t)/2, ε = a mod 2     (ALL a, k — the 2026-08-31
                    unified X/Y-ladder form; reduces to the old
                    κ₀[−1] − κ_{k+1}^sym[−1] − (χ₁κ_{k+1}^anti)[−1] etc.
                    exactly where that was validated)

The p=0 levels follow a unified closed form (dir/top κ-shifts by a parity rule
odd a → −1,−1 / even a → 0,−a, plus a middle ladder s=2..a−1 at shift −a), which
reproduces a1d5 (k=1) and a1d7 (k=2) over ALL seeds — including the former
hold-out, the diameter seed0 `(a=k+1, p=0)`.  k=1 (D₅) and k=2 (D₇) are therefore
FULLY closed-form here.

**The p=1 family is CLOSED for all (a, k) — the unified window rule
(2026-08-31).**  The old "clean form" S(a≥2)={a−1..k} was an unvalidated
extrapolation, REFUTED at every probed `a ≥ 4` (the audit **A65**: at
`(k=3, a=4)` and `(k=4, a=4)` the general-k orthonormality bootstrap
of the source repository and — at k=3 — the direct BPS D₉ fork-quiver trace
agree on `+q⁴χ₁ + q⁶χ₃ + …`, the scaffold-verified ladder law
`(−1)^a q^a χ_p`, while that form starts `−q²χ₁`).  The corrected rule
`−Σ_{s=⌈a/2⌉}^{⌊(v−a)/2⌋} κ_s^anti[−a]` is DERIVED from the exact block
valuations `val(κ_s^anti) = min(4s, 2v−4s)` (verified 20/20 at k ≤ 5): the
leading block is the unique s with val = 2a and admissibility is val ≥ 2a.
It reproduces every previously-validated form identically (the window equals
the old S(a) precisely for a ≤ 3 — why the old form survived so long) and is
validated in the new region at 8 (k,a) points across k = 3, 4, 5, the k=3
case held-out to q²⁴.  Every emission passes the ladder-law guard
(`_to_irrep_guarded`), which converts any future extrapolation failure of
this shape from silent wrong output into a raise.

**The p=0 family is CLOSED too (2026-08-31, second pass) — no seed remains
unpinned.**  The old parity-rule form's fixed s-labels (middle s = 2..a−1,
top s = k+1 at shift −a) coincide with the correct VALUATION labels exactly
where it was validated (a ≤ 2 all k; k ≤ 2) — the same small-k coincidence
mechanism as p=1/A65.  The unified X/Y-ladder form (see `_recipe`) is built
on the measured law val(κ₀ − κ_s^sym) = min(4s, 2v−4s), val(κ_s^sym −
κ_t^sym) = min(4s, 2v−4t): the χ₀-content X-ladder `sym_j` at shifts
−(2j+ε) and the χ₂-content Y-pairs `(sym + χ₁·anti)_{(v−t)/2}` at shifts
−(t+1−ε).  Certificates: 11/11 previously-validated forms reproduced
through q²⁰; the two canonical unique-window k=3 fits ((4,0) exact; (3,0)
via the explicit q³²-shell near-identity `(1+q²+q⁴)χ₁κ₂^anti ≈
−q²χ₁κ₃^anti` at v=9); and 7/7 out-of-sample open seeds at k=4 (q ≤ 18)
and k=5 (q ≤ 12) against the coverage-checked bootstrap.

With both families closed, `a1dodd_layer2` is closed-form at EVERY (a, p,
k); the orthonormality bootstrap of the source repository remains as the
per-k certification instrument (through 𝖖⁴⁰ it reproduces every seed at
k = 1 and k = 2, 2026-09-23), and `A1DoddConeKAlg._trace_residual` calls this
module with no fallback (the one it had could no longer fire).  The
multiply side is closed-form for all k (see `a1dodd_cone_data`).

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
    """Closed-form κ-block recipe for seed (a, p) — defined for ALL (a, k).

    EVERY seed is pinned (2026-08-31) — `_recipe` no longer returns None:
      * p = 1, ALL `a` — the unified WINDOW RULE
        `−Σ_{s=⌈a/2⌉}^{⌊(v−a)/2⌋} κ_s^anti[−a]` (see the in-function
        comment: derived from the block valuations; equals the older
        clean form exactly for a ≤ 3; 8 new (k,a) certificates at a ≥ 4);
      * p = 0, ALL `a` — the unified X/Y-ladder form (see the in-function
        comment: κ₀[−ε] + the sym_j X-ladder + the (sym+χ₁anti) Y-pairs at
        s = (v−t)/2; equals the older parity-rule form exactly where that
        was validated; 2 canonical unique-window fits at k=3 and 7
        out-of-sample certificates at k=4, 5).

    Validation sources: a1d5 / a1d7 (k ≤ 2, complete), the direct BPS
    D-quiver seed traces (k = 3), and the coverage-checked general-k
    orthonormality bootstrap of the source repository (k = 3, 4, 5).
    Beyond the probed (a, k) range both forms are extrapolations of a
    valuation-derived structure, and every emission is cross-checked by the
    ladder-law guard — the A65 lesson applied."""
    if p == 1:
        # THE UNIFIED WINDOW RULE (2026-08-31), for ALL (a, k):
        #
        #     Tr(a, p=1) = -sum_{s = ceil(a/2)}^{floor((v-a)/2)} kappa_s^anti[-a]
        #
        # The window is DERIVED from the block valuations: val(kappa_s^anti)
        # = min(4s, 2v-4s) exactly (verified 20/20 at k = 1..5), the leading
        # block is the unique s with val = 2a (s = a/2 via the 4s branch for
        # even a, s = (v-a)/2 via the 2v-4s branch for odd a), and every
        # other admissible block needs val >= 2a  <=>  ceil(a/2) <= s <=
        # floor((v-a)/2).  For a <= 3 this window COINCIDES with the old
        # clean form's S(a) ({1..k+1} / {a-1..k}) - which is why that form
        # block has val = min(4k, 6) = 6, injecting content at q^{6-a},
        # below the true q^a leading exactly when a >= 4).
        #
        # VALIDATED: reproduces every previously-validated form identically
        # (a <= 3, k <= 5, incl. the shipped a1d5/a1d7); in the new a >= 4
        # region, 8 (k, a) points against the orthonormality bootstrap -
        # (4,1) at k=3 (held-out to q^24 + the direct BPS D_9 window),
        # (4,1)/(5,1) at k=4, (4,1)/(5,1)/(6,1) at k=5 - the k = 1..5
        # top-seed one-termers -kappa_{ceil((k+1)/2)}^anti[-(k+1)] being
        # the degenerate window.  Beyond the probed range every emission is
        # cross-checked by the ladder-law guard in `_to_irrep_guarded` (the
        # check A65's failure mode would have tripped).
        v = 2 * k + 3
        return [("anti", s, -a, False, -1)
                for s in range((a + 1) // 2, (v - a) // 2 + 1)]
    # p == 0 — THE UNIFIED FORM (2026-08-31), for ALL (a, k):
    #
    #     Tr(a, p=0) = kappa0[-eps]
    #                  + sum_{j=1}^{floor(a/2)}  kappa_j^sym[-(2j+eps)]
    #                  - sum_{t=1,3,..,2*ceil(a/2)-1}
    #                        (kappa_s^sym + chi1.kappa_s^anti)[-(t+1-eps)],
    #                        s = (v-t)/2,   eps = a mod 2.
    #
    # The mu-even sector's difference blocks obey the SAME valuation law as
    # the anti sector — val(kappa0 - sym_s) = min(4s, 2v-4s), val(sym_s -
    # sym_t) = min(4s, 2v-4t) (measured across k = 1..5) — so the natural
    # labels are the X-branch (chi0-content at q^{4j}: the sym_j ladder) and
    # the Y-branch (chi2-content at q^{2t}: the (sym + chi1.anti) pairs at
    # s = (v-t)/2).  The OLD parity-rule form used fixed s-labels (middle
    # ladder s = 2..a-1, top s = k+1 at shift -a) which COINCIDE with these
    # valuation labels precisely at a <= 2 (all k) and at k <= 2 — why it
    # validated there and broke at (a >= 3, k >= 3), exactly like the p=1
    # story (A65).
    #
    # VALIDATED: reproduces the previously-validated forms identically
    # (11/11 through q^20: a <= 2 at k <= 5, the k <= 2 diameters); equals
    # the two canonical unique-window fits at k=3 ((3,0) via the explicit
    # q^{32}-shell near-identity, (4,0) exactly); and passes all SEVEN
    # out-of-sample open seeds — (3,0)/(4,0)/(5,0) at k=4 (q<=18) and
    # (3,0)/(4,0)/(5,0)/(6,0) at k=5 (q<=12) — against the coverage-checked
    # bootstrap.  Every emission passes the ladder-law guard.
    v = 2 * k + 3
    eps = a % 2
    rec = [("dir", 0, -eps, False, 1)]
    for j in range(1, a // 2 + 1):
        rec.append(("sym", j, -(2 * j + eps), False, 1))
    for t in range(1, 2 * ((a + 1) // 2), 2):
        s = (v - t) // 2
        sh = -(t + 1 - eps)
        rec.append(("sym", s, sh, False, -1))
        rec.append(("anti", s, sh, True, -1))
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

# ---------------------------------------------------------------------------
# depth-extending memo (2026-09-23)
# ---------------------------------------------------------------------------
# Each closed form is computed once per (k, seed) at the deepest 𝖖-order asked
# so far plus `_MEMO_GROW`, and later calls read a truncation of it.  This is
# exact: every κ block is exact through the order it is built to, so the
# truncation to K of a deeper evaluation IS the evaluation at K (checked
# against the direct evaluation in the suite in the source repository).  A deeper
# call recomputes at the new depth and replaces the entry.  The ladder-law
# guard runs on every evaluation, so a memoised value has passed it at its
# own depth — which is at least K + `_MEMO_GROW`, so the guard also checks
# reads below the lead order `a`, where a direct call at that K skips it.

_MEMO: dict = {}
_MEMO_GROW = 8


def _memo_read(key, K: int, compute) -> dict:
    hit = _MEMO.get(key)
    if hit is None or hit[0] < K:
        depth = max(K, hit[0] if hit is not None else 0) + _MEMO_GROW
        hit = (depth, compute(depth))
        _MEMO[key] = hit
    return {q: dict(d) for q, d in hit[1].items() if q <= K}


def _vacuum_trace_direct(k: int, K: int) -> dict:
    return _to_irrep(_build(k, [("dir", 0, 0, False, 1)], K), K)


def vacuum_trace(k: int, K: int) -> dict:
    """Tr(1) = κ₀, the sl(2)_{-(4k+4)/(2k+3)} vacuum character, to 𝖖-order K.
    Served from the depth-extending memo (`_memo_read`)."""
    return _memo_read(("vac", k), K, lambda depth: _vacuum_trace_direct(k, depth))


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


def _to_irrep_guarded(qmu: dict, K: int, k: int, a: int, p: int) -> dict:
    """`_to_irrep` + the LADDER-LAW GUARD: every elementary seed trace must
    lead with `(−1)^a 𝖖^a χ_p` (the scaffold-verified law,
    a probe in the source repository).  A recipe whose emission
    violates this is wrong — this is exactly the invariant the audit's
    failure mode broke (`−𝖖²χ₁` where `+𝖖⁴χ₁` was true), so the guard turns
    any future recipe-extrapolation failure from silent wrong mathematics
    into an immediate raise.  Skipped when `K < a` (window below the lead)."""
    out = _to_irrep(qmu, K)
    if K >= a:
        lead = min(out) if out else None
        want = {p: (-1) ** a}
        if lead != a or out[a] != want:
            raise RuntimeError(
                f"a1dodd_layer2: LADDER-LAW GUARD tripped for seed (a={a}, "
                f"p={p}, k={k}): leading content {{{lead}: "
                f"{out.get(lead)}}} != {{{a}: {want}}} — the recipe is wrong "
                f"here (cf. the audit); use the bootstrap and report")
    return out


def seed_trace(k: int, idx: int, K: int) -> dict:
    """Elementary trace `Tr(seed_idx)` (idx ∈ {0..2k+1}) to 𝖖-order K.
    Every emission passes the ladder-law guard (`_to_irrep_guarded`).
    Served from the depth-extending memo (`_memo_read`)."""
    a, p = idx_to_ap(k, idx)               # validates idx before memoising
    return _memo_read(("seed", k, idx), K,
                      lambda depth: _seed_trace_direct(k, idx, depth))


def _seed_trace_direct(k: int, idx: int, K: int) -> dict:
    a, p = idx_to_ap(k, idx)
    recipe = _recipe(k, a, p)
    if recipe is None:
        raise NotImplementedError(
            f"a1dodd_layer2: p=0 a={a} (idx={idx}, k={k}) recipe not yet "
            f"pinned; p=0 a<=2 and all p=1 are closed-form (see module "
            f"docstring); the orthonormality bootstrap of the source "
            f"repository computes it")
    return _to_irrep_guarded(_build(k, recipe, K), K, k, a, p)


def seed_trace_ap(k: int, a: int, p: int, K: int) -> dict:
    """Elementary trace by (level a, parity p)."""
    return seed_trace(k, ap_to_idx(k, a, p), K)
