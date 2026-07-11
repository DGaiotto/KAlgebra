"""Spine-free SU(2)+N_f=4 on the **bracelet (Chebyshev) canonical basis**, over
**SO(8)** flavour.

The canonical basis is the classical positive skein **bracelet** basis: for each
curve slope `s` (SL(2,ℤ)/Farey orbit of the three separating curves a,b,c), the
Chebyshev tower

    T₀(s)=𝟙,  T₁(s)=γ_s,  T_{k+1}(s)=γ_s·T_k(s) − T_{k-1}(s),

with the manifestly **positive, few-term** products

    Tⱼ(s)·Tₖ(s) = T_{j+k}(s) + T_{|j-k|}(s)                  (same ray, 2 terms)
    T₁(a)·T₁(b) = 𝖖·T₁(c) + 𝖖⁻¹·T₁(c̃) + 8s·𝟙              (cross ray, 3 terms)

— the cross-ray product IS the bare Kauffman bracket (`T₁=γ`), so all structure
constants are positive; the peripheral `P₀P₃+P₁P₂` is exactly the matter spinor
**8s** of Spin(8) (via the W_ch half-sum frame map, `su2_nf4_so8`).  ρ is the
conformal involution (ρ²=1, trivial on curves, ⋆ on flavour); the basis is
bar-invariant (bar = A→A⁻¹ = 𝖖→𝖖⁻¹ fixes the multicurve labels).

Variable: **A² = 𝖖** (skein over `Z[A^±]=Z[𝖖^{±1/2}]`).  Multiply = intrinsic
skein (`skein_algebra`), re-expressed in the bracelet basis by a triangular peel
(T_k leads with the k-parallel-copy).  NO BPS/RG/torus engine on any path.
"""
from __future__ import annotations

import os
import sys
from math import gcd as _gcd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from skein_algebra import APoly                      # noqa: E402
from su2_nf4_skein_chars import _S, GA, GB, GC, GAT, GBT, GCT     # noqa: E402
from su2_nf4_so8 import periph_x_weights, apoly_to_q, _R8, _AB    # noqa: E402
from zplus_ring import RElement                                   # noqa: E402

_ONE = _S.one()


def _primitive(core):
    g = 0
    for x in core:
        g = _gcd(g, x)
    return (core, 0) if g == 0 else (tuple(x // g for x in core), g)


_T_CACHE: dict[tuple, object] = {}


def bracelet(slope, k: int):
    """T_k(slope) as a skein element (Chebyshev: T₀=1, T₁=γ, T_{k+1}=γT_k−T_{k-1})."""
    slope = tuple(slope)
    if k < 0:
        raise ValueError("k≥0")
    if k == 0:
        return _ONE                       # the identity basis element χ₀=1
    key = (slope, k)
    if key in _T_CACHE:
        return _T_CACHE[key]
    g = _S.curve(slope)
    # SU(2) CHARACTERS χ_n = Chebyshev 2nd kind U_n (χ₀=1, χ₁=γ=W₁ the
    # fundamental Wilson line): χ_{n+1}=γ·χ_n−χ_{n-1}.  a=W₁ generates the SU(2)
    # character cone, with the SU(2) Clebsch χ_n·χ_m = Σ_{l=|n-m|,step2}^{n+m} χ_l
    # (so χ₁²=χ₂+1 = 2⊗2=3+1, NOT the Chebyshev-1st-kind χ₂+2).
    tkm1, tk = _ONE, g                     # χ₀=1, χ₁=γ
    for _ in range(2, k + 1):
        tkp1 = g * tk + (-1) * tkm1
        tkm1, tk = tk, tkp1
    _T_CACHE[key] = (g if k == 1 else tk)
    return _T_CACHE[key]


def _peel_to_bracelet(el) -> dict:
    """Decompose a skein element into the bracelet basis: {(slope,k) |
    (slope,k,periph) | ('P',periph): APoly}.  Triangular (T_k leads with the
    k-parallel-copy of its slope)."""
    work = {lbl: APoly(dict(c._c)) for lbl, c in el.terms.items()}
    out: dict = {}
    while True:
        cand = [l for l in work if any(l[0]) and not work[l].is_zero()]
        if not cand:
            break
        lbl = max(cand, key=lambda v: sum(v[0]))
        core, periph = lbl
        prim, m = _primitive(core)
        coeff = work[lbl]
        key = (prim, m, periph) if any(periph) else (prim, m)
        out[key] = out.get(key, APoly.zero()) + coeff
        for clbl, cc in bracelet(prim, m).terms.items():
            ccore, cperiph = clbl
            tp = tuple(x + y for x, y in zip(periph, cperiph))
            tl = (ccore, tp)
            work[tl] = work.get(tl, APoly.zero()) - coeff * cc
            if work[tl].is_zero():
                work.pop(tl, None)
    for lbl, c in work.items():
        if not c.is_zero():
            out[("P", lbl[1])] = out.get(("P", lbl[1]), APoly.zero()) + c
    return {k: v for k, v in out.items() if not v.is_zero()}


def bracelet_mul(s1, k1, s2, k2) -> dict:
    """Canonical bracelet product T_{k1}(s1)·T_{k2}(s2) (skein, then peel)."""
    return _peel_to_bracelet(bracelet(s1, k1) * bracelet(s2, k2))


def bracelet_mul_so8(s1, k1, s2, k2, channel: str = "c"):
    """T_{k1}(s1)·T_{k2}(s2) over R(Spin(8)) in 𝖖 (A²=𝖖).  Returns
    (curves, flavour) with curves={(slope,k):{q:int}} and flavour a Spin(8)
    RElement-valued {q-power: RElement} on 𝟙 (peripheral → SO(8) via W_ch)."""
    prod = bracelet_mul(s1, k1, s2, k2)
    curves = {}
    flav_x: dict[int, dict] = {}     # q-power -> accumulated x-weight dict
    for lbl, co in prod.items():
        if isinstance(lbl[0], str) and lbl[0] == "P":
            xw = periph_x_weights(lbl[1], channel)
            for e, c in co._c.items():
                if e % 2:
                    raise ValueError("odd A-power in identity flavour")
                d = flav_x.setdefault(e // 2, {})
                for x, m in xw.items():
                    d[x] = d.get(x, 0) + c * m
        else:
            curves[lbl] = apoly_to_q(co)
    flavour = {}
    for q, d in flav_x.items():
        r = _R8.from_abelian(RElement(_AB, d))
        if not r.is_zero():
            flavour[q] = r
    return curves, flavour


__all__ = ["bracelet", "bracelet_mul", "bracelet_mul_so8", "_peel_to_bracelet"]


if __name__ == "__main__":
    NAME = {(0, 0, 0, 0): "1", (1, 0, 0, 0): "8v", (0, 0, 1, 0): "8s",
            (0, 0, 0, 1): "8c", (0, 1, 0, 0): "28"}

    def fmt(r):
        return " + ".join((f"{c}·" if c != 1 else "") + NAME.get(b, str(b))
                          for b, c in sorted(r.terms.items())) or "0"

    print("=== same-ray SU(2) characters: χₙ·χₘ = Σ_{l=|n-m|,step2}^{n+m} χ_l ===")
    for (j, k) in [(1, 1), (1, 2), (2, 2), (2, 3)]:
        got = bracelet_mul(GA, j, GA, k)
        want = {}
        for l in range(abs(j - k), j + k + 1, 2):
            if l == 0:
                want[("P", (0, 0, 0, 0))] = APoly.one()
            else:
                want[(GA, l)] = APoly.one()
        ok = (got == want)
        terms = [(('χ%d' % l[1]) if l[0] != 'P' else '𝟙') for l in got]
        print(f"  χ_{j}·χ_{k} = {terms}  SU(2)-Clebsch={ok}")

    print("\n=== cross-ray over SO(8): T₁ᵃ·T₁ᵇ (positive, matter=8s) ===")
    curves, flav = bracelet_mul_so8(GA, 1, GB, 1)
    cn = {GA: "T₁ᵃ", GB: "T₁ᵇ", GC: "T₁ᶜ", GCT: "T₁ᶜ̃"}
    print("  curves:", {cn.get(l[0], l[0]): v for l, v in curves.items()})
    print("  flavour·𝟙:", {f"𝖖^{q}": fmt(r) for q, r in flav.items()})
    allpos = all(all(v > 0 for v in d.values()) for d in curves.values()) and \
        all(all(m > 0 for m in r.terms.values()) for r in flav.values())
    print(f"  ALL STRUCTURE CONSTANTS POSITIVE: {allpos}")
    print("done")
