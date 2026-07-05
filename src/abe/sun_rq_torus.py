"""SUNRQTorus — the SU(N) Rational Quantum Torus (the trace-zero sibling of URQTorus).

`SU(N) = U(N)` restricted to the **trace-zero magnetic sublattice** `{m⃗ : Σ mᵢ = 0}`
(the gauge group SU(N) ⊂ U(N), the central U(1)/det ungauged → a frozen central
direction).  The trace-zero atoms `U_{m⃗}` (`Σ mᵢ = 0`) carry the SU(N) magnetic
charges; minuscule monopoles are *gone* (no trace-zero fundamental coweight), so the
minimal monopole is the **adjoint** `m⃗ = (1, 0, …, 0, −1)`, which in U(N) is a product
of two minuscules and carries genuine **bubbling** (a magnetic-0 Wilson-sector tail).

This class follows the `URQTorus` design rule — *"a different gauge group carries a
different G-cocycle / measure and is therefore a different class"* (the SU(N) measure
≠ the U(N) measure).  The trace-zero restriction is **closed** under the URQTorus
algebra (multiply: `m⃗ + m⃗' stays trace-zero`; ρ: `m⃗ → −m⃗`), and the U(N) √measure ρ
restricted to the trace-zero sector **is** the SU(N) ρ (the central-U(1) part acts
trivially on trace-zero charges — this is the N=2 fact verified against the certified
`AbelianizedSU2KAlg`, generalised).  So `SUNRQTorus` **inherits** the validated
multiply / ρ / ρ⁻¹ / bar / recognize / well_formed verbatim and overrides **only the
Schur trace measure**:

    Tr_{SU(N)}(f₀)  =  [ trace_v0(center_det(f₀), N, K) ] · (q²;q²)_∞^{−2},

i.e. center the magnetic-0 residual's central (det = v₀⋯v_{N−1}) degree to 0 — a det
degree `d ≢ 0 mod N` is charged under the SU(N) center (non-zero N-ality) and
integrates to 0 — then take the pure-U(N) residue and cancel **one** photon-pair
`(q²;q²)_∞^{−2}` (the decoupled central-U(1) vector multiplet: SU(N) (rank N−1) carries
one fewer `(q²;q²)_∞²` factor than U(N) (rank N) in this normalization).

The N=2 measure was gleaned and validated `==` the BPS pure-SU(2) leg
`pure_ade_kalgebra([("A",1)])`.  The "odd central charge → 0" rule of SU(2)
is the `N = 2` case of the general "`d ≢ 0 mod N → 0`" N-ality rule.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from abelianized_torus import VRational, VLaurent
from laurent_poly import LaurentPoly
from urq_torus import URQTorus
from pure_un_kalgebra import trace_v0


__all__ = ["SUNRQTorus", "center_det_degree", "poch2_inverse_power"]


# ---------------------------------------------------------------------------
# Helpers: central (det = v₀⋯v_{N−1}) centering and the (q²;q²)_∞^{−p} series.
# ---------------------------------------------------------------------------
def _det_degree(f0: VRational) -> int | None:
    """The homogeneous central (det = v₀⋯v_{N−1}) degree of a residual `f0`
    (numerator total v-degree − denominator total v-degree); `None` if `f0` is
    zero.  A canonical's residual is homogeneous in the central direction (det is
    central), so any numerator term gives the degree."""
    if f0.num.is_zero():
        return None
    ve = next(iter(f0.num._terms))            # any term — homogeneous
    num_deg = sum(ve)
    den_deg = sum(f0.den.values())            # each (v_i − q^M v_j) has v-degree 1
    return num_deg - den_deg


def center_det_degree(f0: VRational, N: int):
    """Shift `f0`'s central (det) degree to 0 by multiplying by `det^{−d/N}`
    (`det = v₀⋯v_{N−1}`, i.e. subtract `d/N` from every v-coordinate).

    Returns the centered `VRational`, or ``None`` when the central degree `d` is
    **not divisible by N** — a non-zero N-ality (charged under the SU(N) center
    `Z_N`), whose Schur trace vanishes.  (`N = 2`: `d` odd → 0, the spinor / half-
    integer-spin case.)"""
    d = _det_degree(f0)
    if d is None:
        return f0
    if d % N != 0:
        return None                            # non-zero N-ality → trace 0
    a = d // N
    num = {tuple(x - a for x in ve): lp for ve, lp in f0.num._terms.items()}
    return VRational(VLaurent(num, n=N), dict(f0.den), n=N)


_POCH2_INV: dict = {}


def poch2_inverse_power(p: int, K: int) -> LaurentPoly:
    """`(q²;q²)_∞^{−p}` as a `LaurentPoly` truncated to order `q^K` (p ≥ 0)."""
    key = (p, K)
    if key in _POCH2_INV:
        return _POCH2_INV[key]
    # (q²;q²)_∞ = ∏_{k≥1}(1 − q^{2k}); invert as a power series (constant term 1).
    poch = {0: 1}
    for k in range(1, K // 2 + 1):
        nxt: dict = {}
        for e, c in poch.items():
            nxt[e] = nxt.get(e, 0) + c
            if e + 2 * k <= K:
                nxt[e + 2 * k] = nxt.get(e + 2 * k, 0) - c
        poch = nxt
    inv = {0: 1}
    for n in range(1, K + 1):
        v = -sum(poch.get(j, 0) * inv.get(n - j, 0) for j in range(1, n + 1))
        if v:
            inv[n] = v
    out = LaurentPoly({0: 1})
    base = LaurentPoly(inv)
    for _ in range(p):
        out = LaurentPoly({e: c for e, c in (out * base)._coeffs.items()
                           if 0 <= e <= K})
    _POCH2_INV[key] = out
    return out


# ---------------------------------------------------------------------------
# The SU(N) rational quantum torus.
# ---------------------------------------------------------------------------
class SUNRQTorus(URQTorus):
    """Trace-zero (SU(N)) restriction of `URQTorus(N)`.

    Same residual storage and algebra as `URQTorus` (atoms keyed by the U(N)
    magnetic N-tuple, kept on the trace-zero sublattice `Σ mᵢ = 0`); the **only**
    override is the Schur trace, which uses the SU(N) measure (central-degree
    centering + `(q²;q²)_∞^{−2}` photon cancel)."""

    # ρ is inherited: U(N)'s √measure cocycle restricted to the trace-zero sector
    # IS the SU(N) ρ (the central-U(1) part is trivial on trace-zero charges).

    # internal q-order margin: the U(N) residue's near-cutoff coefficients can
    # be clipped by the N!-divisibility boundary, so compute a little past K and
    # truncate back (an A9-style window margin; empirically 0 already suffices).
    _TRACE_MARGIN = 4

    def _sun_schur(self, f0, K: int) -> LaurentPoly:
        """The SU(N) Schur-measure residue of a magnetic-0 residual `f0`:
        center the central degree, take the pure-U(N) residue `trace_v0`
        (prefactor `(q²;q²)_∞^{2N}`), and cancel ONE photon-pair
        `(q²;q²)_∞^{−2}` → the rank-(N−1) SU(N) prefactor `(q²;q²)_∞^{2(N−1)}`
        (one fewer than U(N))."""
        if f0 is None or f0.is_zero():
            return LaurentPoly.zero()
        centered = center_det_degree(f0, self._N)
        if centered is None:                      # non-zero N-ality → 0
            return LaurentPoly.zero()
        # `adaptive=True` resolves the needed negative-q extent internally (the
        # pure-U(N) `trace`'s discipline) — a deep canonical's low-q terms draw on
        # higher internal orders, so a fixed margin under-resolves the q⁰ at small K
        # (compute exactly, truncate last).  A small extra margin on top guards the
        # N!-divisibility boundary.
        Kint = K + self._TRACE_MARGIN
        bare = trace_v0(centered, self._N, Kint, adaptive=True)
        prod = bare * poch2_inverse_power(2, Kint)
        # Truncate ABOVE K only — negative q-exponents are deliberately KEPT.
        # Clipping them here (the pre-2026-07 behaviour) made the off-span
        # negative-window checks downstream (`orthonormality` below,
        # `KAlgebra.verify_orthonormality`) vacuously true.  No canonical label
        # has ever shown negative trace content (checked pre-filter on the
        # N=2..3 towers), so keeping the window is a pure safety-net revival.
        return LaurentPoly({e: c for e, c in prod._coeffs.items()
                            if e <= K})

    def trace(self, K: int = 8, adaptive: bool = True) -> LaurentPoly:
        """Pure-SU(N) Schur trace — the SU(N) measure residue of the magnetic-0
        residual `f_{(0,…,0)}` (central-degree centered, one U(N) photon cancelled)."""
        return self._sun_schur(self._f.get((0,) * self._N), K)

    def inner(self, other: "SUNRQTorus", K: int = 8) -> LaurentPoly:
        """`⟨a,b⟩ = Tr(ρ(a)·b)` — the §6b block-diagonal pairing (inherited
        `_pairing_f0`) closed with the **SU(N)** trace."""
        if self._f.keys().isdisjoint(other._f.keys()):
            return LaurentPoly.zero()
        f0 = self._pairing_f0(other)
        return self._sun_schur(f0, K)

    def orthonormality(self, other: "SUNRQTorus", K: int = 2):
        """`Tr(ρ(self)·other) = δ + O(q)` via the SU(N) inner product; returns
        ``False`` on negative-q content (off-span), else the `LaurentPoly`.
        (Live guard: `_sun_schur` keeps negative exponents, so this check can
        actually fire — it was vacuous while the trace pre-clipped to `e ≥ 0`.)"""
        t = self.inner(other, K)
        if any(c != 0 for e, c in t._coeffs.items() if e < 0):
            return False
        return t
