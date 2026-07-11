"""Spine-free SU(2)+N_f=4 trace → **Spin(8)** (matter in the **8v**, Wilson
frame).

The trace is the exact **Schur–Askey–Wilson contour functional**
(`schur_measure`, a pure q-series integral — no BPS / RG / torus):

    Tr(χ_k) = ⟨χ_k⟩ = (1/2)[z⁰] (N(z)/D(z)) · χ_k(z),

with `χ_k` the GNO-dual SO(3) spin-k character (the canonical curve tower) and
`χ_half(d)` the d-dim character for the fundamental-Wilson (spinor-class) sector.
The flavour `μ_i` are the SO(8) **orthogonal-frame** fugacities (8v weights
`μ_i^{±1}`), so a moment's weight diagram un-branches to **Spin(8) irreps** via
`Spin8ZPlusRing.from_abelian` (moment weight `m_i` → doubled-e `2 m_i`).

Validated (oracle/measure anchors):
  * `Tr(1) = 1 + 28·q² + (1+28+300)·q⁴ + …`   (28 = SO(8) current adjoint),
  * `Tr(W_½)|q^{1/2} = 8v`                      (the matter, in 8v),
  * `Tr(L_k) = O(q^k)`  and the diagonal orthonormality
    `⟨L_j,L_k⟩ = Σ_{m=|j-k|}^{j+k} Tr(L_m) = δ_{jk} + O(q)`.

**Bootstrap structure (per the N_f=3 template, here simplified by conformality).**
N_f=4 is conformal: `ρ² = id`, so ρ²-cyclicity is plain trace cyclicity
`Tr(xy)=Tr(yx)`.  The anchors are pinned by cyclicity + orthonormality + SO(3)/CG
fusion in the **Spin(8) irrep basis**; the `schur_measure` contour is the exact
cross-check anchor (and the route to the higher q-orders).  The q-grading:
`q_Schur = A⁴` (the by_q key `qh` is the q_Schur half-power; in Coulomb q =
q_Schur^{1/2}, the key `qh` is the q-power).
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from schur_measure import (                      # noqa: E402
    full_measure, moment, chi as _chi_su2, chi_half as _chi_half, by_q,
)
from spin8_characters import Spin8ZPlusRing                   # noqa: E402
from zplus_ring import RElement, AbelianZPlusRing             # noqa: E402

_R8 = Spin8ZPlusRing()
_AB = AbelianZPlusRing(rank=4)

# small-irrep names (Dynkin → label), for readable output
SPIN8_NAME = {
    (0, 0, 0, 0): "1", (1, 0, 0, 0): "8v", (0, 0, 1, 0): "8s", (0, 0, 0, 1): "8c",
    (0, 1, 0, 0): "28", (2, 0, 0, 0): "35v", (0, 0, 2, 0): "35s", (0, 0, 0, 2): "35c",
    (1, 0, 1, 0): "56vs", (1, 0, 0, 1): "56vc", (0, 0, 1, 1): "56sc",
    (1, 1, 0, 0): "160v", (0, 2, 0, 0): "300", (3, 0, 0, 0): "112v",
}

_MEASURE_CACHE: dict[int, dict] = {}


def _measure(qh_max: int) -> dict:
    if qh_max not in _MEASURE_CACHE:
        _MEASURE_CACHE[qh_max] = full_measure(qh_max)
    return _MEASURE_CACHE[qh_max]


def _unbranch(weight_dict) -> RElement:
    """{(m1..m4): int} orthogonal-frame weights → Spin(8) RElement
    (moment weight m_i → doubled-e 2 m_i)."""
    ab = RElement(_AB, {tuple(2 * x for x in w): c for w, c in weight_dict.items()})
    return _R8.from_abelian(ab)


_TRCHI_CACHE: dict[tuple[int, int], dict] = {}


def tr_chi(k: int, qh_max: int = 8) -> dict:
    """Tr(χ_k) of the integer-spin (GNO-dual SO(3)) curve tower, as
    {q-power: Spin8 RElement}.  q-power = the by_q key (Coulomb q)."""
    key = (k, qh_max)
    if key not in _TRCHI_CACHE:
        bq = by_q(moment(_measure(qh_max), _chi_su2(k)))
        _TRCHI_CACHE[key] = {qh: _unbranch(wd) for qh, wd in sorted(bq.items())}
    return _TRCHI_CACHE[key]


_TRWILSON_CACHE: dict[tuple[int, int], dict] = {}


def tr_wilson(dim: int, qh_max: int = 9) -> dict:
    """Tr of the fundamental-Wilson (spinor-class) curve, χ_half(dim)
    (dim-dimensional SU(2) character; dim=2 is spin-½ → 8v matter).

    Cached by `(dim, qh_max)`: the Schur-measure moment + Spin(8) un-branching is
    the dominant cost of `SU2Nf4SampleKAlgebra.trace`, and `trace_element` calls
    it once per label — so the cache turns a per-label recompute into one compute
    per `(level, K)` for the whole pairing table.  The returned dict must be
    treated read-only (callers build fresh dicts for triality images)."""
    key = (dim, qh_max)
    cached = _TRWILSON_CACHE.get(key)
    if cached is None:
        bq = by_q(moment(_measure(qh_max), _chi_half(dim)))
        cached = {qh: _unbranch(wd) for qh, wd in sorted(bq.items())}
        _TRWILSON_CACHE[key] = cached
    return cached


def trace_one(qh_max: int = 8) -> dict:
    """Tr(𝟙) = the Schur index of SU(2)+N_f=4 (Spin(8) per q-level)."""
    return tr_chi(0, qh_max)


def inner_same_ray(j: int, k: int, qh_max: int = 8) -> dict:
    """⟨L_j, L_k⟩ on one ray = Σ_{m=|j-k|}^{j+k} Tr(L_m) (SO(3) CG, ρ²=id so
    bar/ρ act trivially on the ray).  {q-power: Spin8 RElement}."""
    out: dict = {}
    for m in range(abs(j - k), j + k + 1):
        for qh, r in tr_chi(m, qh_max).items():
            out[qh] = out.get(qh, _R8.zero()) + r if qh in out else r
    return {qh: out[qh] for qh in sorted(out) if not out[qh].is_zero()}


def fmt(relt: RElement) -> str:
    if relt.is_zero():
        return "0"
    return " + ".join(
        (f"{c}·" if c != 1 else "") + SPIN8_NAME.get(b, str(b))
        for b, c in sorted(relt.terms.items())
    )


__all__ = [
    "tr_chi", "tr_wilson", "trace_one", "inner_same_ray", "fmt",
    "SPIN8_NAME", "_R8",
]


if __name__ == "__main__":
    print("=== Tr(𝟙) (Schur index, Spin(8) per q-power) ===")
    for qh, r in trace_one().items():
        print(f"  q^{qh}: {fmt(r)}")
    print("\n=== Tr(W_½) (fundamental Wilson; q^1 = matter 8v) ===")
    for qh, r in tr_wilson(2).items():
        print(f"  q^{qh}: {fmt(r)}")
    print("\n=== diagonal orthonormality ⟨L_j,L_k⟩|q⁰ = δ_jk ===")
    ok = True
    for j in range(4):
        for k in range(j, 4):
            ip = inner_same_ray(j, k)
            q0 = ip.get(0, _R8.zero())
            want = _R8.one() if j == k else _R8.zero()
            good = (q0 == want)
            ok = ok and good
            print(f"  ⟨L_{j},L_{k}⟩|q⁰ = {fmt(q0)}  [{'OK' if good else 'FAIL'}]")
    print(f"\nDIAGONAL ORTHONORMALITY: {'ALL OK' if ok else 'FAIL'}")
