"""QuiverURQTorus — the matter-enriched rational quantum torus for linear
U(N₁)×⋯×U(N_n) quivers (bifundamental on each link, optional fundamentals
per node).

Elements live **natively in the f presentation on the product cocharacter
lattice** (user direction 2026-06-11: "same strategy → `U_{m₁,…,m_n}` with
magnetic charge `m_i` at the i-th node"):

    x  =  Σ_{m⃗}  f_{m⃗}(𝔮^{m⃗} v) · U_{m⃗},        f_{m⃗} = Σ_{k⃗} f^{(k⃗)}_{m⃗}·μ^{k⃗},

with `v = (v⁽¹⁾|…|v⁽ⁿ⁾)` the flat product-torus variables (rank `M = ΣN_a`)
and `k⃗ ∈ Z^L` the μ-levels (one slot per link, then per fundamental flavour
— the `quiver_over_pure` slot layout).  `U_{m⃗}` is the quiver theory's own
monopole atom — `(⊗_a U⁽ᵃ⁾_{m_a})` normalized by the √(full measure); the
matter half of that normalization is the per-cell rung product

    Z(m⃗) = ∏_links ∏_{j,l: d_jl<0} ∏_{s<|d_jl|} (1 + μ_e·q^{2s−|d_jl|+1}·v⁽ᵃ⁾_j/v⁽ᵃ⁺¹⁾_l)
            · ∏_nodes ∏_{i≤Nf_a} ∏_{j: m_j<0} ∏_{s<|m_j|} (1 + μ_{a,i}·q^{2s−|m_j|+1}·v⁽ᵃ⁾_j),

`d_jl = m⁽ᵃ⁾_j − m⁽ᵃ⁺¹⁾_l` the link pair-differences — i.e. the certified
U(N)+N_f zero-mode ladder **per (j,l) cell** at rung variable
`x_jl = μ_e·v⁽ᵃ⁾_j/v⁽ᵃ⁺¹⁾_l` (the gauged-flavour structure certified
against the flow in `tests/test_quiver_over_pure.py`).  As in
`MatterURQTorus`, the dressing is **internal bookkeeping** of the
dress/de-dress converters and the cocycles, not part of the API.

The algebra is the same shape as the single-node matter class:

* product: `U_{m⃗}·U_{m⃗′} = R̃_{m⃗,m⃗′}·W_{m⃗,m⃗′}·U_{m⃗+m⃗′}` with `R̃` the
  **block product** of the per-node pure gauge cocycles (denominators from
  gauge only) and `W` the finite matter cocycle — `T_{m⃗′}` shifts each rung
  by `q^{d(m⃗′)}` with `d` linear in `m⃗`, so `W` is the 1-D matter cocycle
  cell-by-cell: net numerator, computed by the same triangular division;
* `ρ` = per-level block GTwist (pure √gauge-measure conjugation) followed by
  the per-image-atom matter factor: division by the (q-free, bar-centered)
  Z-top monomial + the level star `k_slot ↦ −k_slot − D_slot(a⃗)`;
* trace = the **product** Schur-measure residue of the bare magnetic-0
  content with the link/flavour matter factors `M` inserted (single pass);
* `well_formed` = W1 (every `f` component q-palindromic) + W2 (the joint
  q-extreme slice is a single per-node-Levi leading orbit, multiplicity 1,
  at a single level) — the executable "peel until bubbling is O(𝔮)".

The native build (product-and-peel) is the next increment — it needs the
joint-chart `_struct_const` lift; generators (`minuscule`/`wilson` per
node) + the native product + `well_formed` land here.
"""
from __future__ import annotations

import os
import sys
from itertools import permutations

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from abelianized_torus import VRational, VLaurent
from urq_torus import URQTorus, RHO, _shift, _qbar, _Rtilde

from pure_un_closed_form import _levi_decompose
from pure_un_kalgebra import (
    _vinv_vrational, _vdiv, _inv_root, _v0_coeff, _cached_poch2N,
    _schur_measure_euler, _laurent_truncate, _vlaurent_truncate)

from math import factorial


# ===========================================================================
# Shape helpers (ranks = per-node N_a; Nf = per-node fundamental counts;
# links = hyper edges as node-index pairs — default the consecutive chain.
# A self-link (a, a) is ADJOINT matter at node a: rungs on the within-block
# pair-differences, M-factor cells on all N² within-block weights including
# the N diagonal zero-weights).
# ===========================================================================
def _offsets(ranks):
    off = [0]
    for N in ranks:
        off.append(off[-1] + N)
    return tuple(off)


def _chain(n):
    return tuple((a, a + 1) for a in range(n - 1))


def _norm_links(ranks, links):
    """Normalize a link list: `None` = the consecutive chain (the historical
    shape); else a tuple of `(a, b)` node-index pairs, `a == b` allowed
    (adjoint)."""
    if links is None:
        return _chain(len(ranks))
    out = []
    n = len(ranks)
    for a, b in links:
        a, b = int(a), int(b)
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"link ({a},{b}) out of node range")
        out.append((a, b))
    return tuple(out)


def _slots(ranks, Nf, links=None):
    """Number of μ-slots: links first, then flavours grouped by node."""
    return len(_norm_links(ranks, links)) + sum(Nf)


def _flavour_slot(ranks, Nf, a, i, links=None):
    return len(_norm_links(ranks, links)) + sum(Nf[:a]) + i


def _embed_vr(vr: VRational, off: int, M: int) -> VRational:
    """Embed a VRational in `N_a` variables into the flat `M`-variable torus
    at block offset `off`."""
    if getattr(vr, "_sq", None):
        raise NotImplementedError("_embed_vr: sq factors unsupported")
    num = {}
    for ve, lp in vr.num._terms.items():
        ne = [0] * M
        for t, e in enumerate(ve):
            ne[off + t] = e
        num[tuple(ne)] = lp
    den = {}
    for (i, j, m), mult in vr.den.items():
        den[(off + i, off + j, m)] = den.get((off + i, off + j, m), 0) + mult
    return VRational(VLaurent(num, n=M), den, n=M)


def _embed_vl(vl: VLaurent, off: int, M: int) -> VLaurent:
    out = {}
    for ve, lp in vl._terms.items():
        ne = [0] * M
        for t, e in enumerate(ve):
            ne[off + t] = e
        out[tuple(ne)] = lp
    return VLaurent(out, n=M)


def _blocks_of(m, ranks):
    off = _offsets(ranks)
    return [tuple(m[off[a]:off[a + 1]]) for a in range(len(ranks))]


# ===========================================================================
# Dressing cells: (slot, v-weight vector, q-shift) rungs per atom.
# ===========================================================================
def _rungs(atom, ranks, Nf, links=None):
    """The rung list of `Z(m⃗)` at `atom`: `(slot, vw, shift)` per rung —
    links on negative pair-differences `d_jl` (within-block for a self-link,
    where the diagonal `j == l` never rungs since `d = 0`), fundamentals on
    negative `m_j`; `vw` a flat v-exponent vector, shifts bar-centered."""
    links = _norm_links(ranks, links)
    n = len(ranks)
    off = _offsets(ranks)
    M = off[-1]
    out = []
    for e, (a, b) in enumerate(links):
        for j in range(off[a], off[a + 1]):
            for l in range(off[b], off[b + 1]):
                d = atom[j] - atom[l]
                if d < 0:
                    dep = -d
                    vw = tuple((1 if t == j else 0) - (1 if t == l else 0)
                               for t in range(M))
                    for s in range(dep):
                        out.append((e, vw, 2 * s - dep + 1))
    for a in range(n):
        for i in range(Nf[a]):
            slot = _flavour_slot(ranks, Nf, a, i, links)
            for j in range(off[a], off[a + 1]):
                if atom[j] < 0:
                    dep = -atom[j]
                    vw = tuple(1 if t == j else 0 for t in range(M))
                    for s in range(dep):
                        out.append((slot, vw, 2 * s - dep + 1))
    return out


def _rung_levels(atom, ranks, Nf, links=None) -> dict:
    """`Z(m⃗)` per μ-level `{k⃗: VRational}` — the finite bar-centered rung
    expansion (the `_matter_rung_levels` cell generalization)."""
    L = _slots(ranks, Nf, links)
    M = _offsets(ranks)[-1]
    levels = {(0,) * L: {(0,) * M: LaurentPoly({0: 1})}}
    for (slot, vw, sh) in _rungs(atom, ranks, Nf, links):
        out: dict = {}
        for kv, row in levels.items():
            for ve, c in row.items():
                out.setdefault(kv, {}).setdefault(ve, LaurentPoly.zero())
                out[kv][ve] = out[kv][ve] + c
                kv2 = tuple(x + (1 if t == slot else 0)
                            for t, x in enumerate(kv))
                ve2 = tuple(x + w for x, w in zip(ve, vw))
                out.setdefault(kv2, {}).setdefault(ve2, LaurentPoly.zero())
                out[kv2][ve2] = out[kv2][ve2] + c * LaurentPoly({sh: 1})
        levels = out
    return {kv: VRational.from_vlaurent(VLaurent(
        {ve: c for ve, c in row.items() if not c.is_zero()}, n=M))
        for kv, row in levels.items()}


_W_CACHE: dict = {}
_RT_CACHE: dict = {}
_GT_CACHE: dict = {}


def _gauge_rtilde(m, mp, ranks) -> VRational:
    """The block-product gauge cocycle: `R̃_{m⃗,m⃗′} = ∏_a embed(R̃_{m_a,m′_a})`
    (denominators from gauge only — there are no cross-block roots)."""
    key = (tuple(m), tuple(mp), tuple(ranks))
    hit = _RT_CACHE.get(key)
    if hit is None:
        off = _offsets(ranks)
        M = off[-1]
        hit = None
        for a, N in enumerate(ranks):
            blk = _embed_vr(_Rtilde(m[off[a]:off[a + 1]],
                                    mp[off[a]:off[a + 1]], N), off[a], M)
            hit = blk if hit is None else (hit * blk)
        hit = hit.simplify()
        _RT_CACHE[key] = hit
    return hit


def _gauge_gtilde(m, ranks, inverse: bool = False) -> VRational:
    """The block-product √gauge-measure twist `G̃_{m⃗} = ∏_a embed(G̃_{m_a})`."""
    from urq_torus import RHO_INV
    key = (tuple(m), tuple(ranks), inverse)
    hit = _GT_CACHE.get(key)
    if hit is None:
        tw = RHO_INV if inverse else RHO
        off = _offsets(ranks)
        M = off[-1]
        hit = None
        for a, N in enumerate(ranks):
            blk = _embed_vr(tw._Gtilde(m[off[a]:off[a + 1]], N), off[a], M)
            hit = blk if hit is None else (hit * blk)
        hit = hit.simplify()
        _GT_CACHE[key] = hit
    return hit


def _quiver_cocycle(m, mp, ranks, Nf, links=None) -> dict:
    """The matter cocycle `W_{m⃗,m⃗′} = T_{−m⃗′}(Z_{m⃗})·T_{m⃗}(Z_{m⃗′})/Z_{m⃗+m⃗′}`
    per μ-level `{r⃗: VRational}` — finite net numerator (the 1-D matter
    cocycle cell-by-cell), computed once by triangular division and cached."""
    links = _norm_links(ranks, links)
    key = (tuple(m), tuple(mp), tuple(ranks), tuple(Nf), links)
    hit = _W_CACHE.get(key)
    if hit is not None:
        return hit
    M = _offsets(ranks)[-1]
    t = tuple(x + y for x, y in zip(m, mp))
    Zm = _rung_levels(m, ranks, Nf, links)
    Zmp = _rung_levels(mp, ranks, Nf, links)
    Zt = _rung_levels(t, ranks, Nf, links)
    neg_mp = tuple(-x for x in mp)
    num: dict = {}
    for k1, z1 in Zm.items():
        a = _shift(z1, neg_mp, M)
        for k2, z2 in Zmp.items():
            b = _shift(z2, tuple(m), M)
            k = tuple(x + y for x, y in zip(k1, k2))
            term = (a * b).simplify()
            num[k] = term if k not in num else (num[k] + term).simplify()
    budget = max((sum(k) for k in num), default=0) + len(
        _rungs(t, ranks, Nf, links)) + 1
    seen = set(num)
    for k in list(num):
        for dz in Zt:
            seen.add(tuple(x + y for x, y in zip(k, dz)))
    W: dict = {}
    for k in sorted(seen, key=lambda k: (sum(k), k)):
        acc = num.get(k, VRational.from_scalar(LaurentPoly.zero(), n=M))
        for kp, w in W.items():
            dr = tuple(x - y for x, y in zip(k, kp))
            if any(x < 0 for x in dr) or not any(dr):
                continue
            z = Zt.get(dr)
            if z is None:
                continue
            acc = (acc + (w * z * VRational.from_scalar(
                LaurentPoly({0: -1}), n=M))).simplify()
        if not acc.is_zero():
            if sum(k) > budget:
                raise NotImplementedError(
                    f"quiver cocycle W[{m},{mp}]: tail past the rung budget "
                    f"at level {k} — finiteness (net-numerator) violated?")
            W[k] = acc
    _W_CACHE[key] = W
    return W


# ===========================================================================
# Product-measure trace kernel (the `trace_v0` transcription).
# ===========================================================================
_PROD_MEASURE_CACHE: dict = {}


def _prod_measure(ranks, K) -> VLaurent:
    key = (tuple(ranks), K)
    m = _PROD_MEASURE_CACHE.get(key)
    if m is None:
        off = _offsets(ranks)
        M = off[-1]
        m = VLaurent({(0,) * M: LaurentPoly({0: 1})}, n=M)
        for a, N in enumerate(ranks):
            m = _vlaurent_truncate(
                m * _embed_vl(_schur_measure_euler(N, K), off[a], M), K)
        _PROD_MEASURE_CACHE[key] = m
    return m


def _prod_trace_v0(u0, ranks, K: int = 8, adaptive: bool = True) -> LaurentPoly:
    """v-only **product** Schur-measure residue of the magnetic-0 residual:
    `Tr = (q²;q²)_∞^{2M}/∏N_a! · ∮ dv/v · (∏_a measure_a) · u0` — the
    `trace_v0` transcription with the block-product measure and Weyl
    average."""
    if u0 is None:
        return LaurentPoly.zero()
    if u0._sq:
        raise NotImplementedError("quiver trace: sq denominators not handled")
    M = _offsets(ranks)[-1]
    num = {ve: dict(lp._coeffs) for ve, lp in u0._num._terms.items()}
    nz = []
    for (i, j, m), mult in u0._den.items():
        for _ in range(mult):
            if m == 0:
                num = _vdiv(num, i, j, M)
            else:
                nz.append((i, j, m))
    nf = 1
    for N in ranks:
        nf *= factorial(N)

    def core(K_int: int) -> LaurentPoly:
        s = VLaurent({ve: LaurentPoly(lp) for ve, lp in num.items()}, n=M)
        for (i, j, m) in nz:
            s = _vlaurent_truncate(s * _inv_root(i, j, m, M, K_int), K_int)
        v0 = _v0_coeff(s, _prod_measure(ranks, K_int), K_int)
        if v0.is_zero():
            return LaurentPoly.zero()
        qs = _laurent_truncate(v0 * _cached_poch2N(M, K_int), K)
        out = {}
        for e in sorted(qs._coeffs):
            qd, r = divmod(qs._coeffs[e], nf)
            if r != 0:
                break
            if qd:
                out[e] = qd
        return LaurentPoly(out)

    if not adaptive:
        return core(K)
    min_q = 0
    for lp in num.values():
        if lp:
            min_q = min(min_q, min(lp))
    safe = ((K - min_q + 4 + 7) // 8) * 8
    K_int = min(safe, ((K + 4 + 7) // 8) * 8)
    res = core(K_int)
    while K_int < safe:
        K_int = min(safe, K_int + 8)
        nxt = core(K_int)
        if nxt._coeffs == res._coeffs:
            return nxt
        res = nxt
    return res


# ===========================================================================
# Joint (per-node-Levi) recognition for W2.
# ===========================================================================
def _joint_dominance_key(m, ranks):
    return tuple(tuple(sorted(b, reverse=True)) for b in _blocks_of(m, ranks))


def _joint_levi_blocks(m, ranks):
    """Runs of equal value WITHIN each gauge block (global indices) — equal
    `m` values across different nodes must not merge."""
    off = _offsets(ranks)
    blocks = []
    for a in range(len(ranks)):
        start = off[a]
        for i in range(off[a] + 1, off[a + 1] + 1):
            if i == off[a + 1] or m[i] != m[start]:
                blocks.append(list(range(start, i)))
                start = i
    return blocks


def _joint_recognize_leading(f_dict, ranks) -> dict:
    """`recognize_leading` on the product lattice: leading joint Weyl orbit
    (block-wise dominance), Levi-decomposed with the block-refined Levi
    structure.  Returns `{joint_label: LaurentPoly}` with `joint_label` a
    tuple of per-node lower-Kapustin `(m, e)` pairs.

    Dominance-key ties (several atoms of the leading orbit are in support)
    are broken to the **in-frame dominant representative** (block-wise
    descending) — the Levi e-read is only meaningful in that frame.  (The
    pure class dodges this implicitly via constructor insertion order.)"""
    M = _offsets(ranks)[-1]
    if not f_dict:
        return {}
    best = max(_joint_dominance_key(m, ranks) for m in f_dict)
    cands = [m for m in f_dict if _joint_dominance_key(m, ranks) == best]
    dom = [m for m in cands
           if all(b == tuple(sorted(b, reverse=True))
                  for b in _blocks_of(m, ranks))]
    if not dom:
        raise RuntimeError(
            "joint recognize_leading: dominant representative of the leading "
            "orbit not in support")
    m_dom = dom[0]
    c = _shift(f_dict[m_dom], tuple(m_dom), M).simplify()
    if c.den or c._sq:
        raise RuntimeError("joint recognize_leading: leading residual not polynomial")
    blocks = _joint_levi_blocks(m_dom, ranks)
    out = {}
    for e, c_e in _levi_decompose(c.num, blocks, M):
        lab = []
        off = _offsets(ranks)
        for a in range(len(ranks)):
            mb = m_dom[off[a]:off[a + 1]]
            eb = e[off[a]:off[a + 1]]
            lab.append((tuple(reversed(mb)), tuple(reversed(eb))))
        out[tuple(lab)] = c_e
    return out


def _joint_recognize_q_extreme(f_dict, ranks) -> dict:
    """`recognize_q_extreme` on the product lattice (the rank-generic
    val/slice extraction + the joint leading recognizer)."""
    M = _offsets(ranks)[-1]
    vals: dict = {}
    for m, f in f_dict.items():
        f = f.simplify()
        if f.num.is_zero():
            continue
        vnum = min(min(lp._coeffs) for lp in f.num._terms.values())
        vden = sum(Mm * mult for (i, j, Mm), mult in f.den.items() if Mm < 0)
        vals[tuple(m)] = (vnum - vden, f, vnum)
    if not vals:
        return {}
    qext = min(v[0] for v in vals.values())
    sl_f: dict = {}
    for m, (val, f, vnum) in vals.items():
        if val != qext:
            continue
        sign = 1
        shift = [0] * M
        den_lead: dict = {}
        for (i, j, Mm), mult in f.den.items():
            for _ in range(mult):
                if Mm > 0:
                    shift[i] -= 1
                elif Mm == 0:
                    den_lead[(i, j, 0)] = den_lead.get((i, j, 0), 0) + 1
                else:
                    shift[j] -= 1
                    sign = -sign
        acc: dict = {}
        for ve, lp in f.num._terms.items():
            if vnum in lp._coeffs:
                nve = tuple(ve[t] + shift[t] for t in range(M))
                acc[nve] = LaurentPoly({0: sign * lp._coeffs[vnum]})
        sl_f[m] = VRational(VLaurent(acc, n=M), den_lead, n=M)
    out: dict = {}
    for k, c in _joint_recognize_leading(sl_f, ranks).items():
        vv = list(c._coeffs.values())
        out[k] = int(vv[0]) if len(vv) == 1 else int(sum(vv))
    return out


# ===========================================================================
# The class.
# ===========================================================================
class QuiverURQTorus:
    """Quiver-enriched rational quantum torus element: native `f`-residual
    storage `{atom: {k_vec: VRational}}` over the product cocharacter
    lattice; immutable value object.  Shape = `(ranks, Nf)` for the default
    consecutive-chain links, `(ranks, Nf, links)` otherwise (a self-link
    `(a, a)` is adjoint matter at node a — N=2*)."""

    __slots__ = ("_f", "_ranks", "_Nf", "_links", "_M", "_L")

    def __init__(self, f_residuals: dict, ranks, Nf=None, links=None):
        self._ranks = tuple(int(N) for N in ranks)
        n = len(self._ranks)
        self._Nf = tuple(int(x) for x in (Nf if Nf is not None else (0,) * n))
        self._links = _norm_links(self._ranks, links)
        self._M = _offsets(self._ranks)[-1]
        self._L = _slots(self._ranks, self._Nf, self._links)
        f: dict = {}
        for m, row in f_residuals.items():
            keep = {}
            for k, vr in row.items():
                vr = vr.simplify()
                if not vr.is_zero():
                    keep[tuple(k)] = vr
            if keep:
                f[tuple(m)] = keep
        self._f = f

    # ----- constructors / converters ---------------------------------
    @classmethod
    def zero(cls, ranks, Nf=None, links=None) -> "QuiverURQTorus":
        return cls({}, ranks, Nf, links)

    @classmethod
    def one(cls, ranks, Nf=None, links=None) -> "QuiverURQTorus":
        ranks = tuple(int(N) for N in ranks)
        n = len(ranks)
        Nf = tuple(int(x) for x in (Nf if Nf is not None else (0,) * n))
        M = _offsets(ranks)[-1]
        vr = VRational.from_scalar(LaurentPoly({0: 1}), n=M)
        return cls({(0,) * M: {(0,) * _slots(ranks, Nf, links): vr}},
                   ranks, Nf, links)

    @classmethod
    def rg_image_from_pures(cls, xs, ranks, Nf=None, links=None) -> "QuiverURQTorus":
        """**A statement about `RG(a)`, not a native constructor** (the
        `MatterURQTorus.rg_image_from_pure` caveat verbatim): on the joint
        anti-dominant cone the flow image of `⊗_a L_{(m_a,e_a)}` has the
        per-node pure residuals verbatim on `U_{m⃗}` atoms (level 0).
        `xs` = per-node pure `URQTorus` elements.  Cross-check only."""
        ranks = tuple(int(N) for N in ranks)
        off = _offsets(ranks)
        M = off[-1]
        n = len(ranks)
        Nf = tuple(int(x) for x in (Nf if Nf is not None else (0,) * n))
        zero = (0,) * _slots(ranks, Nf, links)
        out: dict = {}

        def rec(a, atom, vr):
            if a == n:
                dst = out.setdefault(tuple(atom), {})
                dst[zero] = vr if zero not in dst else (
                    dst[zero] + vr).simplify()
                return
            for ma, fa in xs[a].residuals().items():
                rec(a + 1, atom + list(ma), (vr * _embed_vr(fa, off[a], M)).simplify())

        rec(0, [], VRational.from_scalar(LaurentPoly({0: 1}), n=M))
        return cls(out, ranks, Nf, links)

    @classmethod
    def minuscule(cls, node: int, m, e, ranks, Nf=None, links=None) -> "QuiverURQTorus":
        """NATIVE dressed-minuscule generator at one node (identity
        elsewhere): atoms = the node's joint Weyl orbit of `(m, e)` embedded
        in the product lattice, residual the monomial `v^{σ(e)}`; level 0."""
        ranks = tuple(int(N) for N in ranks)
        n = len(ranks)
        Nf = tuple(int(x) for x in (Nf if Nf is not None else (0,) * n))
        off = _offsets(ranks)
        M = off[-1]
        N = ranks[node]
        m = tuple(int(x) for x in m)
        e = tuple(int(x) for x in e)
        zero = (0,) * _slots(ranks, Nf, links)
        out: dict = {}
        seen = set()
        for sig in permutations(range(N)):
            am = tuple(m[i] for i in sig)
            ae = tuple(e[i] for i in sig)
            if (am, ae) in seen:
                continue
            seen.add((am, ae))
            atom = [0] * M
            ve = [0] * M
            for t in range(N):
                atom[off[node] + t] = am[t]
                ve[off[node] + t] = ae[t]
            mono = VRational.from_vlaurent(
                VLaurent({tuple(ve): LaurentPoly({0: 1})}, n=M))
            dst = out.setdefault(tuple(atom), {})
            dst[zero] = mono if zero not in dst else (
                dst[zero] + mono).simplify()
        return cls(out, ranks, Nf, links)

    @classmethod
    def wilson(cls, node: int, e, ranks, Nf=None, links=None) -> "QuiverURQTorus":
        """NATIVE Wilson generator: `χ_e(v⁽ⁿᵒᵈᵉ⁾)` at the zero atom."""
        ranks = tuple(int(N) for N in ranks)
        n = len(ranks)
        Nf = tuple(int(x) for x in (Nf if Nf is not None else (0,) * n))
        off = _offsets(ranks)
        M = off[-1]
        sys.path.insert(0, os.path.join(_HERE, "implementations"))
        from un_nf_dressed_generators import _schur_monomials
        terms = {}
        for ve, z in _schur_monomials(tuple(sorted(e, reverse=True))).items():
            ne = [0] * M
            for t, x in enumerate(ve):
                ne[off[node] + t] = x
            terms[tuple(ne)] = LaurentPoly({0: z})
        vr = VRational.from_vlaurent(VLaurent(terms, n=M))
        return cls({(0,) * M: {(0,) * _slots(ranks, Nf, links): vr}},
                   ranks, Nf, links)

    @classmethod
    def from_family(cls, family: dict, ranks, Nf=None, links=None) -> "QuiverURQTorus":
        """De-dress a per-level bare family `{k⃗: {atom: VRational}}` into
        native `f` (triangular inversion of the atom normalization)."""
        ranks = tuple(int(N) for N in ranks)
        n = len(ranks)
        Nf = tuple(int(x) for x in (Nf if Nf is not None else (0,) * n))
        M = _offsets(ranks)[-1]
        atoms = sorted({a for row in family.values() for a in row})
        levels = sorted(family, key=lambda k: (sum(k), k))
        out: dict = {}
        for atom in atoms:
            rungs = _rung_levels(atom, ranks, Nf, links)
            ft: dict = {}
            for k in levels:
                b = family[k].get(atom)
                acc = b if b is not None else VRational.from_scalar(
                    LaurentPoly.zero(), n=M)
                for kp, fv in ft.items():
                    dk = tuple(x - y for x, y in zip(k, kp))
                    if any(x < 0 for x in dk):
                        continue
                    z = rungs.get(dk)
                    if z is None:
                        continue
                    acc = (acc + (fv * z * VRational.from_scalar(
                        LaurentPoly({0: -1}), n=M))).simplify()
                if not acc.is_zero():
                    ft[k] = acc
            if ft:
                out[atom] = ft
        return cls(out, ranks, Nf, links)

    def to_family(self) -> dict:
        """Dress back to per-level bare residual dicts `{k⃗: {atom: VRational}}`."""
        out: dict = {}
        for atom, row in self._f.items():
            rungs = _rung_levels(atom, self._ranks, self._Nf, self._links)
            for k0, fv in row.items():
                for dk, z in rungs.items():
                    k = tuple(x + y for x, y in zip(k0, dk))
                    term = (fv * z).simplify()
                    if term.is_zero():
                        continue
                    dst = out.setdefault(k, {})
                    dst[atom] = term if atom not in dst else (
                        dst[atom] + term).simplify()
        return {k: {a: vr for a, vr in row.items() if not vr.is_zero()}
                for k, row in out.items()
                if any(not vr.is_zero() for vr in row.values())}

    # ----- reads -------------------------------------------------------
    def residuals(self) -> dict:
        return {m: dict(row) for m, row in self._f.items()}

    def support(self):
        return sorted(self._f)

    @property
    def shape(self):
        """`(ranks, Nf)` for the default chain (the historical shape read);
        `(ranks, Nf, links)` when the links are non-default (e.g. adjoint)."""
        if self._links == _chain(len(self._ranks)):
            return (self._ranks, self._Nf)
        return (self._ranks, self._Nf, self._links)

    @property
    def links(self):
        return self._links

    # ----- algebra -----------------------------------------------------
    def __add__(self, other: "QuiverURQTorus") -> "QuiverURQTorus":
        assert self.shape == other.shape
        out = {m: dict(row) for m, row in self._f.items()}
        for m, row in other._f.items():
            dst = out.setdefault(m, {})
            for k, vr in row.items():
                dst[k] = (dst[k] + vr).simplify() if k in dst else vr
        return QuiverURQTorus(out, self._ranks, self._Nf, self._links)

    def __mul__(self, other: "QuiverURQTorus") -> "QuiverURQTorus":
        """The native U-product on the product lattice:

            U_{m⃗}·U_{m⃗′} = R̃_{m⃗,m⃗′}·W_{m⃗,m⃗′}·U_{m⃗+m⃗′},
            f^C_{m⃗+m⃗′} += T_{−m⃗′}(f_{m⃗})·T_{m⃗}(f_{m⃗′})·R̃·W,

        `R̃` the block-product gauge cocycle, `W` the finite cached quiver
        matter cocycle.  No dress/de-dress round trip
        (`_mul_via_dress` retains that route as the cross-check)."""
        assert self.shape == other.shape
        M = self._M
        out: dict = {}
        for m, row1 in self._f.items():
            for mp, row2 in other._f.items():
                t = tuple(x + y for x, y in zip(m, mp))
                W = _quiver_cocycle(m, mp, self._ranks, self._Nf, self._links)
                Rt = _gauge_rtilde(m, mp, self._ranks)
                neg_mp = tuple(-x for x in mp)
                dst = out.setdefault(t, {})
                for k1, f1 in row1.items():
                    a = _shift(f1, neg_mp, M)
                    for k2, f2 in row2.items():
                        b = _shift(f2, tuple(m), M)
                        base = (a * b * Rt).simplify()
                        for r, w in W.items():
                            K = tuple(x + y + z for x, y, z in zip(k1, k2, r))
                            term = (base * w).simplify()
                            dst[K] = term if K not in dst else (
                                dst[K] + term).simplify()
        return QuiverURQTorus(out, self._ranks, self._Nf, self._links)

    def _mul_via_dress(self, other: "QuiverURQTorus") -> "QuiverURQTorus":
        """Dress → per-level bare products (block R̃ only) → de-dress —
        the cross-check of the native cocycle product."""
        assert self.shape == other.shape
        M = self._M
        fam: dict = {}
        for k1, row1 in self.to_family().items():
            for k2, row2 in other.to_family().items():
                k = tuple(x + y for x, y in zip(k1, k2))
                dst = fam.setdefault(k, {})
                for m, f1 in row1.items():
                    for mp, f2 in row2.items():
                        t = tuple(x + y for x, y in zip(m, mp))
                        term = (_shift(f1, tuple(-x for x in mp), M)
                                * _shift(f2, tuple(m), M)
                                * _gauge_rtilde(m, mp, self._ranks)).simplify()
                        dst[t] = term if t not in dst else (
                            dst[t] + term).simplify()
        fam = {k: {a: vr for a, vr in row.items() if not vr.is_zero()}
               for k, row in fam.items()}
        fam = {k: row for k, row in fam.items() if row}
        return QuiverURQTorus.from_family(fam, self._ranks, self._Nf,
                                          self._links)

    def bar(self) -> "QuiverURQTorus":
        """`𝔮 → 𝔮⁻¹` on each `f` component (`v`, atoms, μ⃗ fixed) — W1 frame."""
        return QuiverURQTorus(
            {m: {k: _qbar(vr, self._M) for k, vr in row.items()}
             for m, row in self._f.items()}, self._ranks, self._Nf,
            self._links)

    def well_formed_w1(self) -> bool:
        return self.bar() == self

    def well_formed(self):
        """The full acceptance — "peel until bubbling is O(𝔮)": W1 + the
        cross-level joint q-extreme (single level, single per-node-Levi
        leading orbit, multiplicity 1).  Returns `(joint_label, k⃗)` —
        `joint_label` a tuple of per-node lower-Kapustin `(m, e)` pairs —
        else ``False``.  Same scope as the matter certificate: in-span
        single-target build outputs."""
        if not self.well_formed_w1():
            return False
        vals: dict = {}
        for m, row in self._f.items():
            for k, f in row.items():
                f = f.simplify()
                if f.num.is_zero():
                    continue
                vnum = min(min(lp._coeffs) for lp in f.num._terms.values())
                vden = sum(Mm * mult for (i, j, Mm), mult in f.den.items()
                           if Mm < 0)
                vals[(m, k)] = vnum - vden
        if not vals:
            return False
        qext = min(vals.values())
        levels = {k for (m, k), v in vals.items() if v == qext}
        if len(levels) != 1:
            return False
        (k0,) = levels
        slice_f = {m: row[k0] for m, row in self._f.items()
                   if k0 in row and vals.get((m, k0)) == qext}
        r0 = _joint_recognize_q_extreme(slice_f, self._ranks)
        if len(r0) != 1:
            return False
        (label, c), = r0.items()
        return (label, k0) if c == 1 else False

    def rho(self) -> "QuiverURQTorus":
        """ρ — conjugation by √(full measure): the block GTwist per level
        (magnetic antipode + `v̄` + the gauge `G̃` block product), then the
        matter factor of the **image atom**: division by the q-free Z-top
        monomial `v^{E(a⃗)}` (`E = Σ_cells depth·vw`) and the level star
        `k_slot ↦ −k_slot − D_slot(a⃗)` (`D` = the slot's rung count)."""
        M = self._M
        out: dict = {}
        for m, row in self._f.items():
            a = tuple(-x for x in m)
            gt = _gauge_gtilde(m, self._ranks)
            rungs = _rungs(a, self._ranks, self._Nf, self._links)
            E = [0] * M
            D = [0] * self._L
            for (slot, vw, _sh) in rungs:
                D[slot] += 1
                for t in range(M):
                    E[t] += vw[t]
            mono = None
            if any(E):
                mono = VRational.from_vlaurent(VLaurent(
                    {tuple(-x for x in E): LaurentPoly({0: 1})}, n=M))
            for k, vr in row.items():
                g = (_vinv_vrational(vr, M) * gt).simplify()
                if mono is not None:
                    g = (g * mono).simplify()
                k2 = tuple(-x - d for x, d in zip(k, D))
                dst = out.setdefault(a, {})
                dst[k2] = g if k2 not in dst else (dst[k2] + g).simplify()
        return QuiverURQTorus(out, self._ranks, self._Nf, self._links)

    def rho_inverse(self) -> "QuiverURQTorus":
        """ρ⁻¹ — the exact inverse of `rho`: since ρ = (matter Z-top
        division + level star at the image atom) ∘ (block GTwist), the
        inverse **undoes the matter factor at the source atom first**
        (multiply by `v^{E(a⃗)}`, un-star the levels — the star is its own
        inverse at fixed atom) and then applies the inverse block GTwist.
        Certified by ρ∘ρ⁻¹ = ρ⁻¹∘ρ = id roundtrips."""
        M = self._M
        out: dict = {}
        for a, row in self._f.items():
            m = tuple(-x for x in a)
            gtinv = _gauge_gtilde(a, self._ranks, inverse=True)
            rungs = _rungs(a, self._ranks, self._Nf, self._links)
            E = [0] * M
            D = [0] * self._L
            for (slot, vw, _sh) in rungs:
                D[slot] += 1
                for t in range(M):
                    E[t] += vw[t]
            mono = None
            if any(E):
                mono = VRational.from_vlaurent(VLaurent(
                    {tuple(E): LaurentPoly({0: 1})}, n=M))
            for k2, vr in row.items():
                g = vr if mono is None else (vr * mono).simplify()
                k = tuple(-x - d for x, d in zip(k2, D))
                g = (_vinv_vrational(g, M) * gtinv).simplify()
                dst = out.setdefault(m, {})
                dst[k] = g if k not in dst else (dst[k] + g).simplify()
        return QuiverURQTorus(out, self._ranks, self._Nf, self._links)

    # ----- trace (product measure × link/flavour matter factors) -------
    def trace(self, K: int = 8, W: int = 4, Kq_margin: int = 10) -> dict:
        """μ⃗-refined Schur trace: the link/flavour matter factors inserted
        into the **product** measure residue of the bare magnetic-0 content
        (single pass).  Returns `{μ⃗-level: LaurentPoly}` through `q^K`."""
        from habiro import HabiroElement
        ranks, Nf = self._ranks, self._Nf
        n = len(ranks)
        off = _offsets(ranks)
        M = self._M
        L = self._L
        Kq = K + Kq_margin

        def a_n(nn):
            return HabiroElement.nahm_term((-1) ** nn, nn, [nn]).expand(Kq + W)

        # matter-factor cells: links (j, l) → x = v_j/w_l on the link slot
        # (a self-link runs over ALL within-block ordered pairs including the
        # diagonal j == l — the N zero-weights of the adjoint, vw = 0);
        # flavours (a, i, j) → x = v_j on the flavour slot.  Each cell
        # contributes E(μ x)·E(μ⁻¹/x): ±n on the slot, ±n·vw on the v-weight.
        cells = []
        for e, (aa, bb) in enumerate(self._links):
            for j in range(off[aa], off[aa + 1]):
                for l in range(off[bb], off[bb + 1]):
                    vw = tuple((1 if t == j else 0) - (1 if t == l else 0)
                               for t in range(M))
                    cells.append((e, vw))
        for a in range(n):
            for i in range(Nf[a]):
                slot = _flavour_slot(ranks, Nf, a, i, self._links)
                for j in range(off[a], off[a + 1]):
                    vw = tuple(1 if t == j else 0 for t in range(M))
                    cells.append((slot, vw))

        terms = {((0,) * L, (0,) * M): LaurentPoly({0: 1})}
        for (slot, vw) in cells:
            for sign in (+1, -1):
                new: dict = {}
                for nn in range(0, W + 1):
                    c = a_n(nn)
                    for (lv, ve), z in terms.items():
                        lv2 = tuple(x + (sign * nn if t == slot else 0)
                                    for t, x in enumerate(lv))
                        if sum(abs(x) for x in lv2) > W:
                            continue
                        ve2 = tuple(x + sign * nn * w
                                    for x, w in zip(ve, vw))
                        new[(lv2, ve2)] = new.get(
                            (lv2, ve2), LaurentPoly.zero()) + z * c
                terms = new
        Mfac: dict = {}
        for (lv, ve), z in terms.items():
            if not z.is_zero():
                Mfac.setdefault(lv, {})[ve] = z
        Mfac = {lv: VRational.from_vlaurent(VLaurent(row, n=M))
                for lv, row in Mfac.items()}

        fam = self.to_family()
        out: dict = {}
        for k, row in fam.items():
            f0 = row.get((0,) * M)
            if f0 is None:
                continue
            for nlv, mf in Mfac.items():
                lp = _prod_trace_v0((f0 * mf).simplify(), ranks, K,
                                    adaptive=True)
                if lp.is_zero():
                    continue
                mu = tuple(x + y for x, y in zip(k, nlv))
                out[mu] = out.get(mu, LaurentPoly.zero()) + lp
        return {mu: LaurentPoly({e: c for e, c in lp._coeffs.items()
                                 if 0 <= e <= K})
                for mu, lp in out.items()
                if any(0 <= e <= K and c for e, c in lp._coeffs.items())}

    # ----- equality ----------------------------------------------------
    def __eq__(self, other) -> bool:
        if not isinstance(other, QuiverURQTorus):
            return NotImplemented
        if self.shape != other.shape:
            return False
        keys = set(self._f) | set(other._f)
        for m in keys:
            r1 = self._f.get(m, {})
            r2 = other._f.get(m, {})
            for k in set(r1) | set(r2):
                a = r1.get(k)
                b = r2.get(k)
                if a is None or b is None:
                    return False
                if not (a + b * (-1)).simplify().is_zero():
                    return False
        return True

    def __repr__(self) -> str:
        n_terms = sum(len(r) for r in self._f.values())
        if self._links != _chain(len(self._ranks)):
            return (f"QuiverURQTorus(ranks={self._ranks}, Nf={self._Nf}, "
                    f"links={self._links}, atoms={len(self._f)}, "
                    f"terms={n_terms})")
        return (f"QuiverURQTorus(ranks={self._ranks}, Nf={self._Nf}, "
                f"atoms={len(self._f)}, components={n_terms})")
