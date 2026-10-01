"""Spine-free SU(2)+N_f=4 multiply on the **skein character-ray basis** — the
N_f=4 analogue of the `su2_nf{1,2,3}` H-tower multiply, but built on the
intrinsic Kauffman skein of S²₀,₄ (no BPS / RG / quantum-torus engine on any
path).

N_f=4 is **conformal** (ρ-shift 4−N_f=0): no cones.  The canonical basis is the
set of **character rays** — for each simple-closed-curve slope `s` (an SL(2,ℤ)
orbit of the three tetrahedron separating curves `a,b,c`), the SO(3) spin tower
`χ_k(s)`, `k≥0`, of the GNO-dual SO(3) ('t Hooft line fusing as spin-1).  In
skein content `χ_1(s) = γ_s + 1` and the tower obeys SO(3) Clebsch–Gordan:

    χ_j(s)·χ_k(s) = Σ_{m=|j-k|}^{j+k} χ_m(s)          (same ray, all coeff 1)

Cross-ray products are **few-term** (the Askey–Wilson / DAHA(C∨C₁) shape), e.g.

    χ_1(a)·χ_1(b) = χ_1(a)+χ_1(b)+A²χ_1(c)+A⁻²χ_1(c̃) + (P₀P₃+P₁P₂−A²−1−A⁻²)·𝟙

where `c̃` is the twisted curve and `P_p` the peripheral (puncture-flavour) loops.
The flavour coefficients live in the peripheral-loop ring (per-puncture SU(2),
enhanced to Spin(8) in the trace, `su2_nf4_trace`); the matter is the Spin(8)
**8v** (Wilson frame).  ρ is the conformal **involution** fixing each ray.

Multiply is the intrinsic skein product (`skein_algebra.SkeinAlgebra`,
Kauffman resolve), re-expressed in the χ-basis by a triangular peel (χ_m leads
with the m-parallel-copy multicurve).  Validated against the skein algebra and
the BPS oracle; closure is checked by **associativity**
(the suite in the source repository).

The variable is the skein `A`; the Coulomb-branch `q` enters the trace
(`q_Schur = A⁴`, pinned there).  This module is purely the Z[A^±]-form multiply.
"""
from __future__ import annotations

import os
import sys
from math import gcd as _gcd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "skein_sphere")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from skein_algebra import APoly, SkeinAlgebra      # noqa: E402
from triangulation import Triangulation            # noqa: E402

# The tetrahedral S²₀,₄ chart and its three separating curves (the SL(2,ℤ)
# generators a,b,c; their twisted partners ã,b̃,c̃ appear in cross products).
_TET = Triangulation.tetrahedron_S2_4()
_S = SkeinAlgebra(_TET)
GA = (0, 1, 1, 1, 1, 0)
GB = (1, 0, 1, 1, 0, 1)
GC = (1, 1, 0, 0, 1, 1)
# twisted partners (the q^{∓} curve in each channel product)
GAT = (2, 1, 1, 1, 1, 2)
GBT = (1, 2, 1, 1, 2, 1)
GCT = (1, 1, 2, 2, 1, 1)

NPUNCT = _TET.n_punctures
NEDGE = _TET.n_edges
_ONE = _S.one()


# --------------------------------------------------------------------------
# χ-tower of a slope, as a skein element
# --------------------------------------------------------------------------
def _primitive(core):
    """(primitive slope vector, multiplicity m) with core = m·primitive."""
    g = 0
    for x in core:
        g = _gcd(g, x)
    if g == 0:
        return core, 0
    return tuple(x // g for x in core), g


_CHI_CACHE: dict[tuple[tuple[int, ...], int], object] = {}


def chi(slope, k: int):
    """χ_k(slope) as a skein element: χ_0=1, χ_1=γ+1, SO(3) recursion
    χ_1·χ_m = χ_{m+1}+χ_m+χ_{m-1}."""
    slope = tuple(slope)
    if k < 0:
        raise ValueError("spin k must be ≥ 0")
    if k == 0:
        return _ONE
    key = (slope, k)
    if key in _CHI_CACHE:
        return _CHI_CACHE[key]
    g1 = _S.curve(slope) + _ONE
    ckm1, ck = _ONE, g1
    for _ in range(2, k + 1):
        ckp1 = g1 * ck + (-1) * ck + (-1) * ckm1
        ckm1, ck = ck, ckp1
    _CHI_CACHE[key] = (g1 if k == 1 else ck)
    return _CHI_CACHE[key]


# --------------------------------------------------------------------------
# the χ-basis: a canonical element is a dict {(slope, k): APoly} + a flavour
# (peripheral) part {periph-tuple: APoly} on the identity ray.  We store the
# whole thing as {label: APoly} with label = (slope, k) for k≥1, or
# ("P", periph) for a peripheral-flavour·identity term (k=0).
# --------------------------------------------------------------------------
def _peel_to_chi(el) -> dict:
    """Decompose a skein element into the χ-basis.  Returns {label: APoly}
    where label = (slope, k) (k≥1) or ('P', periph_tuple) for k=0 flavour·𝟙.
    Triangular peel: χ_m leads with the m-parallel-copy of its slope."""
    work = {lbl: APoly(dict(c._c)) for lbl, c in el.terms.items()}
    out: dict = {}

    def core_rank(lbl):
        core, _ = lbl
        return sum(core)

    while True:
        # highest non-peripheral core (max total parallel count)
        cand = [lbl for lbl in work if any(lbl[0]) and not work[lbl].is_zero()]
        if not cand:
            break
        lbl = max(cand, key=core_rank)
        core, periph = lbl
        prim, m = _primitive(core)
        coeff = work[lbl]
        # this slope carries χ_m at multiplicity `coeff`, with the SAME periph
        out_key = (prim, m)
        if any(periph):
            # peripheral-dressed curve: keep the periph on the label
            out_key = (prim, m, periph)
        out[out_key] = out.get(out_key, APoly.zero()) + coeff
        # subtract coeff · χ_m(prim) (with the periph carried along)
        chim = chi(prim, m)
        for clbl, cc in chim.terms.items():
            ccore, cperiph = clbl
            tot_periph = tuple(x + y for x, y in zip(periph, cperiph))
            tlbl = (ccore, tot_periph)
            work[tlbl] = work.get(tlbl, APoly.zero()) - coeff * cc
            if work[tlbl].is_zero():
                work.pop(tlbl, None)
    # leftover: peripheral-only (k=0 flavour·identity)
    for lbl, c in work.items():
        if c.is_zero():
            continue
        _core, periph = lbl
        out[("P", periph)] = out.get(("P", periph), APoly.zero()) + c
    return {k: v for k, v in out.items() if not v.is_zero()}


def chi_mul(slope1, k1, slope2, k2) -> dict:
    """Canonical character-ray product χ_{k1}(slope1)·χ_{k2}(slope2), as a
    χ-basis dict {label: APoly} (label = (slope,k) | (slope,k,periph) |
    ('P',periph))."""
    prod = chi(slope1, k1) * chi(slope2, k2)
    return _peel_to_chi(prod)


# --------------------------------------------------------------------------
# ρ — the conformal involution (fixes every character ray)
# --------------------------------------------------------------------------
def rho_chi(slope, k):
    """ρ on a character-ray canonical: the conformal (ρ²=id) involution fixes
    it.  Returns (slope, k)."""
    return (tuple(slope), k)


__all__ = [
    "GA", "GB", "GC", "GAT", "GBT", "GCT",
    "chi", "chi_mul", "rho_chi", "_peel_to_chi", "_S",
]


if __name__ == "__main__":
    # self-tests
    print("=== same-ray SO(3) Clebsch–Gordan ===")
    for (j, k) in [(1, 1), (1, 2), (2, 2)]:
        got = chi_mul(GA, j, GA, k)
        want = {(GA, m): APoly.one() for m in range(abs(j - k), j + k + 1)
                if m >= 1}
        want_identity = abs(j - k) == 0
        # collect: each (GA,m) coeff 1, plus identity if |j-k|=0
        ok = all(got.get((GA, m)) == APoly.one()
                 for m in range(max(1, abs(j - k)), j + k + 1))
        idok = True
        if want_identity:
            idok = got.get(("P", (0, 0, 0, 0))) == APoly.one()
        print(f"  χ_{j}^a·χ_{k}^a: terms={len(got)}  SO(3)={ok and idok}")

    print("\n=== cross-ray χ_1^a·χ_1^b (few-term) ===")
    cross = chi_mul(GA, 1, GB, 1)
    for lbl in sorted(cross, key=lambda x: str(x)):
        print(f"   {lbl}: {cross[lbl]!r}")
    print(f"  #terms = {len(cross)}")

    print("\n=== associativity (χ_1^a·χ_1^a)·χ_1^b == χ_1^a·(χ_1^a·χ_1^b) ===")
    # via skein elements (associativity is automatic in the skein, but check
    # the χ-peel is consistent)
    lhs = (chi(GA, 1) * chi(GA, 1)) * chi(GB, 1)
    rhs = chi(GA, 1) * (chi(GA, 1) * chi(GB, 1))
    print(f"  skein associativity: {lhs == rhs}")
    print("\nself-tests done")
