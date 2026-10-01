"""Reliable, memory-light su2 elementary-trace generator (improved bootstrap).

The su2 entries of the finite zoo ([A₁,D₃]/[A₁,D₅]/[A₁,D₇] = sl(2)₋₄/₃ /
sl(2)₋₈/₅ / sl(2)₋₁₂/₇) get their elementary traces from the orthonormality
bootstrap.  The original `su2_bootstrap.generate_su2` (a) takes Tr(1) from the
BPS-backed `_vacuum_rps`, and (b) falls back to a *growing general-pair pool*
when pure powers don't pin every seed — which both OOMs at high K and produced
a frozen a1d5 table that is **wrong from ~q¹⁹** (under-resolved high-spin tail).

This module fixes both:

  * **Tr(1) = the closed-form Kac–Wakimoto vacuum character** of sl(2) at the
    admissible level `k+2 = 2/v` (`char_fug`, base ρ) — instant, no BPS;
  * **pin every seed by pure powers** `Tr(seed^a)=O(qᵃ)` (a up to `amax`),
    clearing the cone reduce-cache between reductions so memory stays bounded —
    a *small* pool, no general-pair explosion.

For a1d5 this pins all 4 seeds (`free=[]`) and reproduces the
`a1d5_layer2` closed-form characters exactly (validated to q²⁴), while the
frozen table diverges from q¹⁹.  `v` per theory: D_{2k+1} ↦ v = 2k+1.

Public: `vacuum_char(v, K)` and `reliable_seeds(short_id, K)`.
"""
from __future__ import annotations

from fractions import Fraction as Fr

# v per su2 short_id  (D_{2k+1} = sl(2) at k+2 = 2/(2k+1))
_V_OF = {"a1d3": 3, "a1d5": 5, "a1d7": 7}

# ---------------------------------------------------------------------------
# Kac–Wakimoto vacuum character in fugacity (μ-Laurent), N/D, base ρ
# ---------------------------------------------------------------------------
_RHO = 1


def _theta_fug(step, factor, Kmax, base):
    out: dict = {}
    B = Kmax + 8
    f = Fr(factor)
    base = Fr(base)
    for n in range(-B, B + 1):
        beta = 2 * step * n
        qpow = Fr(base) * beta / 2 + f / 2 * (Fr(beta) * beta / 2)
        if qpow != int(qpow) or not (0 <= int(qpow) <= Kmax):
            continue
        qp = int(qpow)
        fin0 = base + f * Fr(beta)
        for sign, w in ((1, 1), (-1, -1)):
            wf = w * fin0
            d = out.setdefault(qp, {})
            d[wf] = d.get(wf, 0) + sign
    return {q: {w: c for w, c in d.items() if c} for q, d in out.items()}


def _zmul(a, b):
    out: dict = {}
    for wa, ca in a.items():
        for wb, cb in b.items():
            out[wa + wb] = out.get(wa + wb, 0) + ca * cb
    return {w: v for w, v in out.items() if v}


def _zsub(a, b):
    out = dict(a)
    for w, c in b.items():
        out[w] = out.get(w, 0) - c
    return {w: v for w, v in out.items() if v}


def _div_by_D0(R):
    """Divide a μ→μ⁻¹ antisymmetric Laurent by (μ − μ⁻¹), exact."""
    rem = dict(R)
    quot: dict = {}
    guard = 0
    while any(c for c in rem.values()):
        guard += 1
        if guard > 20000:
            raise RuntimeError("char_fug: division did not terminate")
        pmax = max(p for p, c in rem.items() if c)
        c = rem[pmax]
        quot[pmax - 1] = quot.get(pmax - 1, 0) + c
        rem[pmax] = rem.get(pmax, 0) - c
        rem[pmax - 2] = rem.get(pmax - 2, 0) + c
        rem = {p: cc for p, cc in rem.items() if cc}
    return {p: c for p, c in quot.items() if c}


def _fug_to_irrep(mud):
    """Weyl-symmetric integer-μ Laurent → {SU(2) irrep n: c}."""
    md: dict = {}
    for p, c in mud.items():
        if Fr(p).denominator != 1:
            raise RuntimeError("char_fug: fractional μ in vacuum")
        md[int(p)] = md.get(int(p), 0) + c
    out: dict = {}
    while any(c for c in md.values()):
        mx = max(p for p, c in md.items() if c)
        if mx < 0:
            raise RuntimeError("char_fug: non-symmetric vacuum")
        c = md[mx]
        out[mx] = out.get(mx, 0) + c
        for k in range(mx, -mx - 1, -2):
            md[k] = md.get(k, 0) - c
        md = {p: cc for p, cc in md.items() if cc}
    return {n: c for n, c in out.items() if c}


def vacuum_char(v: int, K: int, u: int = 2, hv: int = 2) -> dict:
    """Tr(1) for sl(2) at k+2=u/v as {𝖖-power: {irrep n: int}} (q_paper=𝖖²)."""
    Kg = K // 2 + 1
    N = _theta_fug(v, Fr(u, v), Kg, _RHO)
    D = _theta_fug(1, hv, Kg, _RHO)
    D0 = D[0]
    ch_z: dict = {}
    out: dict = {}
    for n in range(0, Kg + 1):
        Rn = dict(N.get(n, {}))
        for j in range(0, n):
            if j in ch_z and (n - j) in D:
                Rn = _zsub(Rn, _zmul(ch_z[j], D[n - j]))
        ch_z[n] = _div_by_D0(Rn) if Rn else {}
        if 2 * n <= K and ch_z[n]:
            out[2 * n] = _fug_to_irrep(ch_z[n])
    return out


# ---------------------------------------------------------------------------
# reliable seed generation: char_fug Tr(1) + pure-power pinning (small pool)
# ---------------------------------------------------------------------------

def reliable_seeds(short_id: str, K: int, *, amax: int = 10,
                   margin: int = 2, verbose: bool = False) -> dict:
    """`{seed_idx: {𝖖-power: {irrep n: int}}}` for a su2 entry, BPS-free and
    memory-light.  Raises if a seed is left unpinned (caller can widen `amax`).

    Reliable to 𝖖^K (unlike the frozen tables' under-resolved tail)."""
    if short_id not in _V_OF:
        raise ValueError(f"reliable_seeds: {short_id!r} is not a su2 D-odd entry")
    import gc
    from su2_bootstrap import _to_pool, _sweep
    from regen import _load_standalone
    from trace_uniqueness_proofs import seed_set, seed_reduction

    mod, prefix = _load_standalone(short_id)
    A = getattr(mod, prefix + "KAlgebra")() if hasattr(mod, prefix + "KAlgebra") \
        else _std_algebra(short_id)
    ident = A.identity()
    seedlabs = [s for s in seed_set(A) if s != ident]
    pos = {sl: p for p, sl in enumerate(seedlabs)}
    idxs = [sl[0][0] for sl in seedlabs]
    Ki = K + margin

    Tr1 = vacuum_char(_V_OF[short_id], Ki)
    pool: list = []
    cd = A.cone_data()
    for idx in idxs:
        for a in range(2, amax + 1):
            try:
                _to_pool(pool, seed_reduction(A, ((idx, a),)), False, ident, pos)
            except Exception:
                break
            # Clear the cone reduce-cache after EACH reduction (not just per
            # seed): the char-basis sub-word cache balloons across a power
            # tower (a=2..amax), so clearing per-power is what keeps memory
            # bounded for the bigger D-odd algebras (a1d7: 7-node, 6 seeds —
            # OOMs without this; bounded + reliable with it).
            if hasattr(cd, "_reduce_word_cache"):
                cd._reduce_word_cache.clear()
            gc.collect()
        if verbose:
            print(f"[{short_id}] idx{idx} reduced (pool={len(pool)})", flush=True)
    Tr, free, status = _sweep(pool, Tr1, Ki, strict=False)
    if free:
        raise RuntimeError(
            f"reliable_seeds({short_id}): seeds {sorted(free)} unpinned at "
            f"amax={amax} (status={status}); widen amax")
    out: dict = {}
    for j, idx in enumerate(idxs):
        out[idx] = {q: Tr[(j, q)] for q in range(1, K + 1) if Tr.get((j, q))}
    return out


def _std_algebra(short_id: str):
    from regen import _load_standalone
    import importlib
    if short_id == "a1d5":
        return importlib.import_module("finite_a1d5_kalg").FiniteA1D5KAlgebra()
    if short_id == "a1d3":
        return importlib.import_module("finite_a1d3_kalg").FiniteA1D3KAlgebra()
    if short_id == "a1d7":
        return importlib.import_module("finite_a1d7_kalg").FiniteA1D7KAlgebra()
    raise ValueError(short_id)
