"""`PinnedPentagonKAlg` — the pentagon extracted from the PINNED
quantum torus.

The sibling `SkeinPentagonKAlg` extracts the pentagon from the stated
engine by localizing at the side arcs (Ore set, token bookkeeping,
common denominators, peel).  On the pinned tier the same construction
is FREE: in `PinnedPolygon(5)` (the quantum torus over
Gamma~ = Gamma_A2 (+) Z^5, `pinned_polygon.py`) every side arc's
quantum trace is a SINGLE MONOMIAL — a unit of the torus — so the
dressed chords

    T~_p = lq^{x_p + 2} · F(D_p) · s_{p+2}^{w} s_{p+4}^{w} s_{p+3}^{-w}

(the same unpin dressing words `w_p` and gauge `x_p` fitted by
`SkeinPentagonAlgebra.unpin()`) are honest
torus elements, computed by plain monomial arithmetic.  Measured:

* every T~_p has boundary block ZERO — the dressing exactly cancels
  the pin, so the dressed chords live on the internal sublattice
  Gamma_A2 (the dressing IS the definition of the pinned canonicals);
* the pentagon relations hold EXACTLY in the pinned torus:
  T~_p T~_{p+1} = 1 + lq^2 T~_{p+3}  (= 1 + q_chart^{-1} T~ at
  q_chart = lq^{-2}), all p in Z_5;
* every T~_p commutes with every side monomial — the factorization
  Pinned = Pentagon (x) T(sides) holds at the PINNED level, no
  localization needed;
* Newton dictionary T~_p <-> L_{g_{p+1}} (the kappa dictionary's
  one-unit shift).

Labels are the BPS chart labels (Z^2 canonical charges).  A canonical
label gamma = a·h_p + b·h_{p+2} (a, b >= 0; h_p = g_{p+1} the T~-frame
of the rho-orbit) forwards to K(gamma) = lq^{-ab} T~_p^a T~_{p+2}^b.
`multiply` is computed GENUINELY on the pinned side: the torus product
of the two K's, peeled against the K-images of the BPS-hinted candidate
labels (labels only; every coefficient comes from the pinned torus) —
the same unique-label peel + per-label normalization (delta) table as
the sibling, with the delta consistency ASSERTED on every reoccurrence.
q-conventions: q_chart = lq^{-2} (the measured dictionary).

`rho` / `rho_inverse` / `trace` are transported from the BPS
realization (documented, as in the sibling); the stated-side trace
is not computed intrinsically.  `build_iso()` returns the
`KAlgebraIso` to the BPS realization (identity label map; verifying it
tests the PINNED structure constants against the chart's —
non-circular).  Registered as the ``'skein-pinned'`` realization of
the pentagon.
"""

from __future__ import annotations

from fractions import Fraction

import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kalgebra import Element, KAlgebra
from laurent_poly import LaurentPoly
from zplus_ring import RPowerSeries, TrivialZPlusRing

from half_laurent import HalfLaurent
from pinned_polygon import PinnedPolygon
from skein_pentagon import SkeinPentagonAlgebra
from skein_pentagon_kalg import ORBIT, SkeinPentagonKAlg

Q = HalfLaurent.monomial


def _f5(x):
    return x % 5


class PinnedPentagonKAlg(KAlgebra):
    """The pinned-torus realization of the pentagon, on chart labels."""

    def __init__(self, bps=None):
        self.pp = PinnedPolygon(5)
        pp = self.pp
        self._sides = [pp.arc_F(r, _f5(r + 1), 1, 1) for r in range(5)]
        self._sides_inv = [pp.inverse_monomial(s) for s in self._sides]
        self._chords = [pp.arc_F(p, _f5(p + 2), 1, 1) for p in range(5)]
        up = SkeinPentagonAlgebra().unpin()
        self._w = up["w"]
        self._x = up["x"]
        self._T = [self._dress(p) for p in range(5)]
        # the T~-frame: T~_p <-> chart label h_p = g_{p+1} (measured)
        self._h = [ORBIT[_f5(p + 1)] for p in range(5)]
        if bps is None:
            from bps_kalgebra import BPSKAlgebra
            bps = BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                              node_charges=[(1, 0), (0, 1)])
        self._bps = bps
        self._mult_cache = {}
        self._fwd_cache = {}
        self._delta = {}

    # ---- the dressed chords ------------------------------------------------

    def _dress(self, p: int) -> dict:
        """T~_p = lq^{x_p+2} F(D_p) · (per the unpin word w_p, ascending
        side order with sign — the sibling's NO convention)."""
        pp = self.pp
        out = pp.scale(self._chords[p], Q(Fraction(self._x[p]) + 2))
        for r in range(5):
            e = -self._w[p][r]
            for _ in range(abs(e)):
                out = pp.multiply(
                    out, self._sides[r] if e > 0 else self._sides_inv[r])
        return out

    def T(self, p: int) -> dict:
        """The dressed chord T~_p (a pin-0 pinned-torus element)."""
        return self._T[_f5(p)]

    def side(self, r: int) -> dict:
        """The pinned side monomial F(s_r(+,+))."""
        return self._sides[_f5(r)]

    # ---- forward map (canonical label -> pinned element) --------------------

    def _cone_decompose(self, gamma):
        for p in range(5):
            g1, g2 = self._h[p], self._h[_f5(p + 2)]
            det = g1[0] * g2[1] - g1[1] * g2[0]
            a = (gamma[0] * g2[1] - gamma[1] * g2[0]) * det
            b = (g1[0] * gamma[1] - g1[1] * gamma[0]) * det
            if det * det == 1 and a >= 0 and b >= 0:
                return p, a, b
        raise ValueError(f"label {gamma} not in any cone")

    def forward(self, gamma) -> dict:
        """K(gamma) = lq^{-ab} T~_p^a T~_{p+2}^b for
        gamma = a h_p + b h_{p+2}."""
        gamma = tuple(gamma)
        if gamma not in self._fwd_cache:
            p, a, b = self._cone_decompose(gamma)
            pp = self.pp
            out = pp.scale(pp.one(), Q(Fraction(-a * b)))
            for _ in range(a):
                out = pp.multiply(out, self._T[p])
            for _ in range(b):
                out = pp.multiply(out, self._T[_f5(p + 2)])
            self._fwd_cache[gamma] = out
        return self._fwd_cache[gamma]

    # ---- KAlgebra primitives ------------------------------------------------

    def coefficient_ring(self):
        return TrivialZPlusRing()

    def identity(self):
        return (0, 0)

    def multiply(self, a, b) -> Element:
        a, b = tuple(a), tuple(b)
        key = (a, b)
        if key in self._mult_cache:
            return self._mult_cache[key]
        pp = self.pp
        prod = pp.multiply(self.forward(a), self.forward(b))
        # candidates from the BPS expansion (LABELS only; coefficients
        # below all come from the pinned torus)
        hint = self._bps.multiply(a, b)
        pool = {tuple(lam): self.forward(tuple(lam)) for lam in hint.terms}
        rem = dict(prod)
        skein_co = {}
        while rem:
            hit = None
            for u in sorted(rem):
                owners = [lam for lam, c in pool.items() if u in c]
                if len(owners) == 1:
                    hit = (u, owners[0])
                    break
            assert hit is not None, f"peel stuck for {a}*{b}"
            u, lam = hit
            r = SkeinPentagonKAlg._mono_ratio_hl(rem[u], pool[lam][u])
            assert r is not None, f"non-monomial peel at {lam}"
            e, sg = r
            piece = pp.scale(pool.pop(lam), Q(e) * sg)
            rem = pp.add(rem, {v: -c for v, c in piece.items()})
            skein_co[lam] = (e, sg)
        # normalization bookkeeping (the sibling's delta scheme): the
        # pinned coefficient must equal lq^{delta(a)+delta(b)-delta(lam)}
        # x c_chart(lam)|_{q->lq^-2}; delta fitted once per label and
        # ASSERTED on every later occurrence.
        out = {}
        da = self._delta_of(a)
        db = self._delta_of(b)
        for lam, (e, sg) in skein_co.items():
            cch = hint.terms[lam]
            ch_hl = SkeinPentagonKAlg._chart_to_hl(cch)
            rel = SkeinPentagonKAlg._mono_ratio_hl(Q(e) * sg, ch_hl)
            assert rel is not None and rel[1] == 1, \
                f"pinned/chart coefficient shapes differ at {lam}"
            want_delta = da + db - rel[0]
            lam = tuple(lam)
            if lam in self._delta:
                assert self._delta[lam] == want_delta, (
                    f"normalization inconsistency at {lam}: "
                    f"{self._delta[lam]} vs {want_delta}")
            else:
                self._delta[lam] = want_delta
            out[lam] = cch
        el = Element({lam: c for lam, c in out.items()})
        self._mult_cache[key] = el
        return el

    def _delta_of(self, lam):
        lam = tuple(lam)
        if lam == (0, 0):
            return Fraction(0)
        if lam in [tuple(g) for g in ORBIT]:
            return Fraction(0)
        if lam not in self._delta:
            raise KeyError(
                f"normalization of {lam} not yet determined; multiply "
                f"generator pairs first (the delta table is built "
                f"incrementally)")
        return self._delta[lam]

    def rho(self, a):
        r = self._bps.rho(tuple(a))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return tuple(r)

    def rho_inverse(self, a):
        r = self._bps.rho_inverse(tuple(a))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return tuple(r)

    def trace(self, a, K: int = 20) -> RPowerSeries:
        return self._bps.trace(tuple(a), K)

    def r_label_decompose(self, label):
        """Trivial flavour-lift coordinate `(label, χ₀)`: unflavoured
        (odd marks)."""
        return tuple(label), self.coefficient_ring().one_basis()

    def r_label_compose(self, section, r_basis_label):
        return tuple(section)

    # ---- the iso witness -----------------------------------------------------

    def build_iso(self):
        from kalgebra_iso import KAlgebraIso
        unit = LaurentPoly({0: 1})

        def fwd(lbl):
            return Element({tuple(lbl): unit})

        def inv(lbl):
            return Element({tuple(lbl): unit})

        return KAlgebraIso(self, self._bps, fwd, inv,
                           name="pentagon[pinned→bps]")
