"""`SkeinPentagonKAlg` — the pentagon as a `KAlgebra` contract
instance realized by the stated-SKEIN engine, via the UNPIN
construction.

Realization
-----------
Labels are the BPS pentagon chart labels (Z², canonical charges).
The five generator charges are the ρ-orbit g₀=(1,0), g₁=(−1,−1),
g₂=(0,1), g₃=(0,−1), g₄=(−1,0); the compatible (q-commuting) pairs
are {g_p, g_{p+2}}, whose spans are the five cluster cones covering
Z².  The forward realization of a canonical label γ = a·g_p +
b·g_{p+2} (a, b ≥ 0) is the localized stated element

    K(γ) = q^{-ab} · T'_p{}^a · T'_{p+2}{}^b,
    T'_p = lq^{x_p+2} · D_p(+,+) · (s_{p+2} s_{p+4} s_{p+3}^{-1})^{-1}

with (x_p) the unpin gauge — the dressed chords of the pinned
pentagon, in the localization at the side arcs (an Ore set; all the
commutation data engine-measured).  `multiply(a, b)` is computed
GENUINELY ON THE SKEIN SIDE: the localized engine product of the two
K's, peeled against the K-images of candidate labels (the BPS
expansion supplies only the candidate LABEL SET; every coefficient
comes from the skein engine).  A nonzero peel remainder raises.

q-conventions: the engine variable lq satisfies q_chart = lq^{-2}
(the same dictionary as the bigon O_Q(SL₂) at Q = q² and the unpin
invariant −2 ↔ −1); structure constants are asserted to have even
lq-exponents and are returned in chart-q `LaurentPoly`s.

Transported primitives (documented): `rho` /
`rho_inverse` (the label rotation — geometrically certified at
generator level: ρ = the one-unit
rotation of the irregular puncture) and `trace` (via the BPS
realization; the intrinsic stated-side trace = the half-index, a
separate work item).  `coefficient_ring` is Trivial (odd marks: no
flavour).

The `KAlgebraIso` to the BPS realization is `build_iso()`
(identity label correspondence; verifying it tests the SKEIN
structure constants against the chart's — non-circular).
Registered as the ``'skein'`` realization of the pentagon object.
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
from skein_pentagon import SkeinPentagonAlgebra

Q = HalfLaurent.monomial

ORBIT = [(1, 0), (-1, -1), (0, 1), (0, -1), (-1, 0)]


def _f5(x):
    return x % 5


class SkeinPentagonKAlg(KAlgebra):
    """The stated-skein realization of the pentagon, on chart labels."""

    def __init__(self, bps=None):
        self.A = SkeinPentagonAlgebra()
        self.P = self.A.P
        up = self.A.unpin()
        self._w = up["w"]
        self._x = up["x"]
        self._Mss = up["Mss"]
        self._Msd = up["Msd"]
        if bps is None:
            from bps_kalgebra import BPSKAlgebra
            bps = BPSKAlgebra(pairing=[[0, 1], [-1, 0]],
                              node_charges=[(1, 0), (0, 1)])
        self._bps = bps
        self._mult_cache = {}
        self._fwd_cache = {}
        self._delta = {}

    # ---- token-level localized arithmetic -------------------------------
    # token: ('D', p) | ('s', r) | ('si', r)

    def _swap_cost(self, r, tok):
        """cost c in  s_r · tok = q^c · tok · s_r."""
        if tok[0] == "D":
            return self._Msd(r, tok[1])
        if tok[0] == "s":
            return self._Mss(r, tok[1])
        return -self._Mss(r, tok[1])

    def _normalize(self, terms):
        """terms: list of (qexp, tokens).  Bubble all 'si' right,
        evaluate the positive prefix in the engine, and combine on a
        common denominator.  Returns (PolyElement, denomvec)."""
        prepared = []
        for qe, toks in terms:
            toks = list(toks)
            changed = True
            while changed:
                changed = False
                for i in range(len(toks) - 1):
                    if toks[i][0] == "si" and toks[i + 1][0] != "si":
                        r = toks[i][1]
                        qe = qe - self._swap_cost(r, toks[i + 1])
                        toks[i], toks[i + 1] = toks[i + 1], toks[i]
                        changed = True
            cut = next((i for i, t in enumerate(toks) if t[0] == "si"),
                       len(toks))
            pos, neg = toks[:cut], toks[cut:]
            el = self.P.one()
            for t in pos:
                if t[0] == "D":
                    el = el * self.A.chord(t[1])
                else:
                    el = el * self.A.side(t[1])
            den = [0] * 5
            # invert the 'si' run: (si_{r1} si_{r2} ...) with the run
            # representing s_{r1}^{-1} s_{r2}^{-1} ... = (… s_{r2}
            # s_{r1})^{-1}; convert the run to the sorted NO(w)^{-1}
            # collecting q-costs.
            run = [t[1] for t in neg]
            # cost of rewriting s_{r1}^{-1}..s_{rk}^{-1} as
            # q^{-C} NO(w)^{-1}: equivalent to sorting the reversed
            # positive word run_rev into NO order.
            rev = list(reversed(run))
            ce = 0
            srt = []
            for r in rev:
                j = len(srt)
                while j > 0 and srt[j - 1] > r:
                    ce += self._Mss(srt[j - 1], r)
                    j -= 1
                srt.insert(j, r)
            qe = qe - ce
            for r in run:
                den[r] += 1
            prepared.append((qe, el, tuple(den)))
        W = tuple(max(p[2][r] for p in prepared) if prepared else 0
                  for r in range(5))
        total = None
        for qe, el, den in prepared:
            u = tuple(W[r] - den[r] for r in range(5))
            # x·NO(den)^{-1} = q^{-C(den,u)} (x·NO(u))·NO(W)^{-1}
            cdu = sum(den[r] * u[t] * self._Mss(r, t)
                      for t in range(5) for r in range(t + 1, 5))
            ext = el
            for r in range(5):
                for _ in range(u[r]):
                    ext = ext * self.A.side(r)
            ext = ext * Q(Fraction(qe) - cdu)
            total = ext if total is None else total + ext
        return total, W

    def _cone_decompose(self, gamma):
        for p in range(5):
            g1, g2 = ORBIT[p], ORBIT[_f5(p + 2)]
            det = g1[0] * g2[1] - g1[1] * g2[0]
            a = (gamma[0] * g2[1] - gamma[1] * g2[0]) * det
            b = (g1[0] * gamma[1] - g1[1] * gamma[0]) * det
            if det * det == 1 and a >= 0 and b >= 0:
                return p, a, b
        raise ValueError(f"label {gamma} not in any cone")

    def _Tword(self, p):
        """tokens + q-exponent of T'_p = q^{x_p+2} D_p NO(-w_p)
        (normalized: relations T'T' = 1 + q^2 T').  NO of a mixed
        exponent vector = ascending-r order with sign per entry."""
        toks = [("D", p)]
        wp = self._w[p]
        for r in range(5):
            e = -wp[r]
            for _ in range(abs(e)):
                toks.append(("s", r) if e > 0 else ("si", r))
        return Fraction(self._x[p] + 2), tuple(toks)

    def _forward_terms(self, gamma):
        p, a, b = self._cone_decompose(tuple(gamma))
        qe = Fraction(-a * b)
        toks = []
        e1, t1 = self._Tword(p)
        e2, t2 = self._Tword(_f5(p + 2))
        for _ in range(a):
            qe += e1
            toks.extend(t1)
        for _ in range(b):
            qe += e2
            toks.extend(t2)
        return [(qe, tuple(toks))]

    def forward(self, gamma):
        """The normalized localized stated element of the canonical
        label γ: (PolyElement numerator, denominator vector)."""
        gamma = tuple(gamma)
        if gamma not in self._fwd_cache:
            self._fwd_cache[gamma] = self._normalize(
                self._forward_terms(gamma))
        return self._fwd_cache[gamma]

    # ---- KAlgebra primitives --------------------------------------------

    def coefficient_ring(self):
        return TrivialZPlusRing()

    def identity(self):
        return (0, 0)

    @staticmethod
    def _chart_to_hl(c: LaurentPoly) -> HalfLaurent:
        """q_chart^k -> lq^{-2k} (the measured dictionary)."""
        out = {}
        for k, v in c._coeffs.items():
            out[Fraction(-2 * k)] = out.get(Fraction(-2 * k), 0) + v
        return HalfLaurent(out)

    @staticmethod
    def _mono_ratio_hl(a: HalfLaurent, b: HalfLaurent):
        """a == (sign) lq^e b ?  -> (e, sign) or None."""
        ia, ib = a.items(), b.items()
        if len(ia) != len(ib):
            return None
        shifts = set()
        for (ea, va), (eb, vb) in zip(ia, ib):
            if vb == 0 or va % vb:
                return None
            sg = va // vb
            if sg not in (1, -1):
                return None
            shifts.add((ea - eb, sg))
            if len(shifts) > 1:
                return None
        return next(iter(shifts))

    def multiply(self, a, b) -> Element:
        a, b = tuple(a), tuple(b)
        key = (a, b)
        if key in self._mult_cache:
            return self._mult_cache[key]
        ta = self._forward_terms(a)
        tb = self._forward_terms(b)
        prod = [(qa + qb, ka + kb) for (qa, ka) in ta for (qb, kb) in tb]
        num, W = self._normalize(prod)
        # candidates from the BPS expansion (labels only)
        hint = self._bps.multiply(a, b)
        cands = {}
        denoms = {}
        for lam in hint.terms:
            n2, W2 = self.forward(tuple(lam))
            cands[tuple(lam)] = (n2, W2)
        # bring everything to a common denominator
        Wc = tuple(max([W[r]] + [w2[r] for (_n2, w2) in cands.values()])
                   for r in range(5))

        def extend(el, w):
            u = tuple(Wc[r] - w[r] for r in range(5))
            cdu = sum(w[r] * u[t] * self._Mss(r, t)
                      for t in range(5) for r in range(t + 1, 5))
            ext = el
            for r in range(5):
                for _ in range(u[r]):
                    ext = ext * self.A.side(r)
            return ext * Q(Fraction(-cdu))

        rem = extend(num, W)
        cmn = {lam: extend(n2, w2) for lam, (n2, w2) in cands.items()}
        # unique-label peel: skein coefficients in lq
        skein_co = {}
        pool = dict(cmn)
        while rem.terms:
            hit = None
            for lbl in sorted(rem.terms):
                owners = [n for n, c in pool.items() if lbl in c.terms]
                if len(owners) == 1:
                    hit = (lbl, owners[0])
                    break
            assert hit is not None, f"peel stuck for {a}*{b}"
            lbl, lam = hit
            r = self._mono_ratio_hl(rem.terms[lbl], pool[lam].terms[lbl])
            assert r is not None, f"non-monomial peel at {lam}"
            e, sg = r
            skein_co[lam] = (e, sg)
            rem = rem - pool[lam] * Q(e) * sg
            pool.pop(lam)
        # normalization bookkeeping: c_skein(lam) must equal
        # lq^{delta(a)+delta(b)-delta(lam)} * c_chart(lam)|_{q->lq^-2};
        # delta is a global per-label table (generators and identity 0),
        # fitted once and ASSERTED on every later occurrence -- the
        # consistency across products is the genuine verification.
        out = {}
        da = self._delta_of(a)
        db = self._delta_of(b)
        for lam, (e, sg) in skein_co.items():
            cch = hint.terms[lam]
            ch_hl = self._chart_to_hl(cch)
            rel = self._mono_ratio_hl(Q(e) * sg, ch_hl)
            assert rel is not None and rel[1] == 1, \
                f"skein/chart coefficient shapes differ at {lam}"
            xshift = rel[0]
            want_delta = da + db - xshift
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

    def _label_section_decompose(self, label):
        """Trivial flavour-lift coordinate — the pentagon skein chart is
        unflavoured (`TrivialZPlusRing`).  Used by `to_R_form`."""
        return (tuple(label), self.coefficient_ring().one())

    def r_label_decompose(self, label):
        """Trivial flavour-lift coordinate `(label, χ₀)`: the pentagon skein
        chart is unflavoured.  Implemented directly (independent of
        `_label_section_decompose`), so `forget()` / ring-hom flavour
        promotion read this coordinate."""
        return tuple(label), self.coefficient_ring().one_basis()

    def r_label_compose(self, section, r_basis_label):
        return tuple(section)

    # ---- the iso witness --------------------------------------------------

    def build_iso(self):
        from kalgebra_iso import KAlgebraIso
        unit = LaurentPoly({0: 1})

        def fwd(lbl):
            return Element({tuple(lbl): unit})

        def inv(lbl):
            return Element({tuple(lbl): unit})

        return KAlgebraIso(self, self._bps, fwd, inv,
                           name="pentagon[skein→bps]")
