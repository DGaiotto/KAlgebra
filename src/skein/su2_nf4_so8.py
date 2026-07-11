"""SO(8)-covariant flavour of the SU(2)+N_f=4 cross-ray multiply — the
correct frame for the skein character-ray product (the algebra has
SO(8) flavour symmetry).

Two flavour frames:
  * the **per-puncture** frame `e_p` — where the peripheral loops `P_p` live
    (`P_p = χ₁(SU(2)_p)`, the puncture doublet, weights `±e_p`);
  * the **SO(8) orthogonal** frame `x_i` — where the matter 8v has weights `±x_i`
    (the `schur_measure` / trace frame).
They differ by the **NON-Weyl half-sum map** `W_ch` (`x = (e_p±e_q)/2`, the
channel's puncture pairing).  Naively using `e_p` as `x_i` "manufactures a
spurious spinor defect" — so flavour coefficients must be pushed through `W_ch`
before un-branching to Spin(8) (`spin8_characters.from_abelian`).

Result (frame-independent across the three channels):

    a·b = a + b + 𝖖·c + 𝖖⁻¹·c̃ + ( 8s − 𝖖 − 1 − 𝖖⁻¹ )·𝟙          over R(Spin(8))

— the matter surfaces as the **spinor 8s** (the 't Hooft-line screening class;
the Wilson sector carries 8v, dyonic 8c — triality).

**Variable:** the Kauffman `A` is pinned to the Coulomb-branch 𝖖 by **A² = 𝖖**
(the skein is over `Z[A^±] = Z[𝖖^{±1/2}]`); closed-curve products give integer
𝖖-powers, the half-integer Wilson sector gives 𝖖^{1/2}.

**Bar invariance** (the K_𝖖 bar axiom): bar = `A→A⁻¹` (= `𝖖→𝖖⁻¹`) fixes every
multicurve/character-ray label, and the SO(8) flavour coefficient
`8s − 𝖖 − 1 − 𝖖⁻¹` is itself bar-symmetric; so the χ-basis is bar-invariant and
`bar(a·b) = b·a` (antimultiplicative).  `verify_bar_invariance` checks it.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from fractions import Fraction
from itertools import product as _iproduct
from math import comb as _comb

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from skein_algebra import APoly                      # noqa: E402
from spin8_characters import Spin8ZPlusRing                       # noqa: E402
from zplus_ring import RElement, AbelianZPlusRing                 # noqa: E402
from su2_nf4_skein_chars import chi_mul, GA, GB, GC               # noqa: E402

_R8 = Spin8ZPlusRing()
_AB = AbelianZPlusRing(rank=4)

# the three puncture pairings (channels)
PAIRINGS = {"a": ((0, 1), (2, 3)), "b": ((0, 2), (1, 3)), "c": ((0, 3), (1, 2))}


# --------------------------------------------------------------------------
# W_ch : per-puncture e-frame  ->  SO(8) orthogonal x-frame  (half-sum map)
# --------------------------------------------------------------------------
def _w_rows(pairing):
    rows = []
    for (p, q) in pairing:
        r1 = [Fraction(0)] * 4
        r2 = [Fraction(0)] * 4
        r1[p], r1[q] = Fraction(1, 2), Fraction(1, 2)
        r2[p], r2[q] = Fraction(1, 2), Fraction(-1, 2)
        rows += [r1, r2]
    return rows


def w_ch(pairing, weight):
    """Map a per-puncture e-weight (4-tuple) to the SO(8) orthogonal x-frame."""
    rows = _w_rows(pairing)
    return tuple(sum(Fraction(weight[i]) * rows[i][p] for i in range(4))
                 for p in range(4))


def _periph_e_weights(k):
    """(k0,k1,k2,k3) peripheral loops -> {per-puncture e-weight: mult}
    (parallel copies: ∏_p (μ_p+μ_p⁻¹)^{k_p})."""
    per = []
    for kp in k:
        wd = Counter()
        for j in range(kp + 1):
            wd[kp - 2 * j] += _comb(kp, j)
        per.append(wd)
    out = Counter()
    for combo in _iproduct(*[w.items() for w in per]):
        w = tuple(e for e, _ in combo)
        m = 1
        for _, c in combo:
            m *= c
        out[w] += m
    return out


def periph_x_weights(periph, channel: str) -> Counter:
    """{SO(8) orthogonal x-weight (doubled-e int 4-tuple): mult} for a
    peripheral-loop label (k0..k3), pushed through the channel's W_ch frame.
    (A single label is generally only PART of a Spin(8) character — accumulate
    the whole flavour before un-branching.)"""
    out = Counter()
    for w, m in _periph_e_weights(periph).items():
        x = w_ch(PAIRINGS[channel], w)
        out[tuple(int(2 * xi) for xi in x)] += m
    return out


def periph_to_spin8(periph, channel: str) -> RElement:
    """SO(8) character of a (single) peripheral-loop label — only valid when the
    label's weight content is itself a genuine character."""
    return _R8.from_abelian(RElement(_AB, dict(periph_x_weights(periph, channel))))


# --------------------------------------------------------------------------
# A  <->  𝖖   (A² = 𝖖)
# --------------------------------------------------------------------------
def apoly_to_q(co: APoly) -> dict:
    """Convert an A-coefficient to a 𝖖-Laurent dict {q-power: int}, A²=𝖖.
    Only even A-powers occur in closed-curve products (→ integer 𝖖-powers)."""
    out = {}
    for e, c in co._c.items():
        if e % 2:
            raise ValueError(f"odd A-power A^{e}: half-integer 𝖖 (Wilson sector)")
        out[e // 2] = out.get(e // 2, 0) + c
    return out


# --------------------------------------------------------------------------
# the SO(8)-covariant cross-ray product
# --------------------------------------------------------------------------
def cross_ray_so8(s1, k1, s2, k2, channel: str = "c"):
    """χ_{k1}(s1)·χ_{k2}(s2) over R(Spin(8)), in 𝖖 (A²=𝖖).
    Returns (curves, flavour_identity):
      * curves  = {(slope,k): {q-power: int}}        — 𝖖-coeff curve terms,
      * flavour_identity = Spin(8) RElement-valued {q-power: RElement} on 𝟙,
    flavour pushed through the channel's W_ch frame map."""
    prod = chi_mul(tuple(s1), k1, tuple(s2), k2)
    curves = {}
    flav_by_q: dict[int, RElement] = {}
    for lbl, co in prod.items():
        if isinstance(lbl[0], str) and lbl[0] == "P":
            periph = lbl[1]
            s8 = periph_to_spin8(periph, channel)
            for e, c in co._c.items():
                if e % 2:
                    raise ValueError("odd A-power in identity flavour")
                qp = e // 2
                flav_by_q[qp] = (flav_by_q.get(qp, _R8.zero())
                                 + (s8 if c == 1 else _scale(s8, c)))
        else:
            curves[lbl] = apoly_to_q(co)
    return curves, {q: r for q, r in flav_by_q.items() if not r.is_zero()}


def _scale(relt: RElement, c: int) -> RElement:
    return RElement(relt.ring, {b: c * m for b, m in relt.terms.items()})


def verify_bar_invariance(s1, k1, s2, k2) -> bool:
    """bar(χ_{k1}(s1)·χ_{k2}(s2)) == χ_{k2}(s2)·χ_{k1}(s1) (antimultiplicative,
    bar = A→A⁻¹ fixing labels)."""
    from su2_nf4_skein_chars import chi
    x, y = chi(tuple(s1), k1), chi(tuple(s2), k2)
    return (x * y).bar() == y.bar() * x.bar()


__all__ = ["PAIRINGS", "w_ch", "periph_to_spin8", "apoly_to_q",
           "cross_ray_so8", "verify_bar_invariance", "_R8"]


if __name__ == "__main__":
    NAME = {(0, 0, 0, 0): "1", (1, 0, 0, 0): "8v", (0, 0, 1, 0): "8s",
            (0, 0, 0, 1): "8c", (0, 1, 0, 0): "28"}

    def fmt(r):
        return " + ".join((f"{c}·" if c != 1 else "") + NAME.get(b, str(b))
                          for b, c in sorted(r.terms.items())) or "0"

    print("=== A² = 𝖖 ;  a·b over R(Spin(8)) ===")
    for ch in ("a", "b", "c"):
        curves, flav = cross_ray_so8(GA, 1, GB, 1, channel=ch)
        fl = " + ".join(f"({fmt(r)})·𝖖^{q}" for q, r in sorted(flav.items()))
        print(f"  [channel {ch} W_ch]  identity flavour: {fl}")
    curves, _ = cross_ray_so8(GA, 1, GB, 1)
    print(f"  curve terms (over 𝖖): "
          f"{ {('a' if l[0]==GA else 'b' if l[0]==GB else 'c' if l[0]==GC else 'c~'): v for l, v in curves.items()} }")

    print("\n=== bar invariance ===")
    for (s1, k1, s2, k2, nm) in [(GA, 1, GB, 1, "a·b"), (GA, 1, GA, 2, "a·χ₂ᵃ"),
                                 (GB, 1, GC, 1, "b·c")]:
        print(f"  bar({nm}) = (rev): {verify_bar_invariance(s1, k1, s2, k2)}")
    print("done")
