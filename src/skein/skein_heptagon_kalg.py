"""`SkeinHeptagonKAlg` — the [A₁,A₄] heptagon as a `KAlgebra` contract
instance realized by the stated-SKEIN engine, via the UNPIN
construction.

The 7-gon is the [A₁,A₄] irregular puncture — RANK 4, the A₄
Argyres–Douglas chart, with TWO chord families (the p+2 "short"
diagonals D2_p and the p+3 "long" diagonals D3_p, 14 chords) and the
Y-system periodicity n = h+2 = 7.  It is ODD-marked, so it carries no
flavour and unpins CLEANLY (no gauge unit, no even-cycle anomaly):
the pentagon story at rank 4.

Realization
-----------
Labels are the intrinsic `A1A2kKAlg(2)` labels ``((a,i,1)…)`` (cone
monomials in the 14 chord generators (1,i) = shorts, (2,i) = longs).
The forward realization sends each generator to its dressed chord in
the localization at the side arcs:

    (1,p)  ->  T(D2_p) = D2_p . NO(-w2_p)
    (2,p)  ->  T(D3_{p-a3}) = D3_{p-a3} . NO(-w3_{p-a3})

with the unpin dressing words w2_0 = (0,0,1,-1,1,-1,1),
w3_0 = (0,-2,0,-1,-1,-1,-1) (rest by rotation)
and the dictionary offset a3 (the short↔long rotation, fixed by the
exchange D2_p D2_{p+1} = unit + long).  A general canonical maps to
the ordered product of its cone generators' words, the cone phase read
off the `A1A2kKAlg(2)` engine.

`multiply(a, b)` is computed GENUINELY ON THE SKEIN SIDE: the
localized engine product of the two forward images, peeled against the
forward images of the intrinsic product's labels (the intrinsic
supplies only the candidate LABEL SET), with a per-label normalization
(δ) table whose generator values are solved from the generator-pair
products and asserted globally consistent — the same mechanism as
`SkeinU1HexagonKAlg`.  δ is a normalization scalar per label, not a
basis element (the returned Elements carry the intrinsic canonical
coefficients), so the canonical-building rule is untouched.

Transported primitives (pending the intrinsic stated-side
trace/half-index): `rho` / `rho_inverse` / `trace` via `A1A2kKAlg(2)`.
`coefficient_ring` is Trivial (odd marks: no flavour).

The `KAlgebraIso` to the intrinsic is `build_iso()` (identity label
correspondence).  Registered as the ``'skein'`` realization of
the heptagon `KAlgebraObject`.
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

from half_laurent import HalfLaurent
from stated_polygon import StatedPolygon

Q = HalfLaurent.monomial
N = 7


def _f7(x):
    return x % 7


def _mss(r, t):
    if t == _f7(r + 1):
        return -1
    if t == _f7(r - 1):
        return 1
    return 0


_M2 = (1, -1, 1, 0, 0, 0, -1)
_M3 = (1, 0, -1, 1, 0, 0, -1)


def _msd2(r, p):
    return _M2[_f7(r - p)]


def _msd3(r, p):
    return _M3[_f7(r - p)]


# unpin dressing words, CENTERED mod the all-ones / CF direction
# (the 7-cycle Mss kernel) to minimise Σ|w| — the un-centered long
# word (0,-2,0,-1,-1,-1,-1) blows the dressed long up to 1489 terms
# AND offsets the exchange by a central CF power; centering to
# (1,-1,1,0,0,0,0) gives a 10-term long and a clean A₄ exchange.
_W2_0 = (0, 0, 1, -1, 1, -1, 1)
_W3_0 = (1, -1, 1, 0, 0, 0, 0)


def _w2(p):
    return tuple(_W2_0[_f7(t - p)] for t in range(N))


def _w3(p):
    return tuple(_W3_0[_f7(t - p)] for t in range(N))


def _no_tokens(wv):
    toks = []
    for r in range(N):
        e = wv[r]
        for _ in range(abs(e)):
            toks.append(("s", r) if e > 0 else ("si", r))
    return toks


def _ts(p):
    """dressed short T(D2_p) tokens."""
    p = _f7(p)
    return [("Ds", p)] + _no_tokens([-x for x in _w2(p)])


def _tl(p):
    """dressed long T(D3_p) tokens."""
    p = _f7(p)
    return [("Dl", p)] + _no_tokens([-x for x in _w3(p)])


def _solve_exact(rows, rhs):
    """Exact Fraction Gaussian elimination.  Returns
    (consistent, solution-with-free-vars-0)."""
    if not rows:
        return True, []
    nv = len(rows[0])
    aug = [r[:] + [v] for r, v in zip(rows, rhs)]
    pivots = []
    rr = 0
    for col in range(nv):
        piv = next((i for i in range(rr, len(aug)) if aug[i][col] != 0),
                   None)
        if piv is None:
            continue
        aug[rr], aug[piv] = aug[piv], aug[rr]
        inv = aug[rr][col]
        aug[rr] = [x / inv for x in aug[rr]]
        for i in range(len(aug)):
            if i != rr and aug[i][col] != 0:
                f = aug[i][col]
                aug[i] = [a - f * b for a, b in zip(aug[i], aug[rr])]
        pivots.append(col)
        rr += 1
    consistent = all(any(x != 0 for x in row[:-1]) or row[-1] == 0
                     for row in aug)
    sol = [Fraction(0)] * nv
    for i, col in enumerate(pivots):
        sol[col] = aug[i][-1]
    return consistent, sol


class SkeinHeptagonKAlg(KAlgebra):
    """The stated-skein realization of the [A₁,A₄] heptagon, on the
    `A1A2kKAlg(2)` labels."""

    # q_int = lq^{ORI} (the bigon dictionary Q = q²; q_chart = lq^{-2})
    ORI = -2

    def __init__(self, intrinsic=None):
        if intrinsic is None:
            from a1a2k_kalg import A1A2kKAlg
            intrinsic = A1A2kKAlg(2)
        self._intr = intrinsic
        self._cd = intrinsic.cone_data()
        self.P = StatedPolygon(N)
        self.Ds = {p: self.P.arc(p, _f7(p + 2), 1, 1) for p in range(N)}
        self.Dl = {p: self.P.arc(p, _f7(p + 3), 1, 1) for p in range(N)}
        self.S = {r: self.P.arc(r, _f7(r + 1), 1, 1) for r in range(N)}
        # dictionary: D2_p ↔ (1,p) (a2 = 0, a free ρ-gauge), D3_p ↔
        # (2,p) (a3 = 0; the long D3_0 IS the canonical (2,0), pinned by
        # the exchange T(D2_0)T(D2_1) = unit + T(D3_0)).  short→(1,·),
        # long→(2,·) is forced by the exchange direction.
        self._a3 = 0
        self._fwd_cache = {}
        self._mult_cache = {}
        self._peel_cache = {}
        self._delta = None

    # ---- token-level localized arithmetic ------------------------------

    def _swap_cost(self, r, tok):
        if tok[0] == "Ds":
            return _msd2(r, tok[1])
        if tok[0] == "Dl":
            return _msd3(r, tok[1])
        if tok[0] == "s":
            return _mss(r, tok[1])
        return -_mss(r, tok[1])

    def _normalize(self, terms):
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
                if t[0] == "Ds":
                    el = el * self.Ds[t[1]]
                elif t[0] == "Dl":
                    el = el * self.Dl[t[1]]
                else:
                    el = el * self.S[t[1]]
            den = [0] * N
            run = [t[1] for t in neg]
            rev = list(reversed(run))
            ce = 0
            srt = []
            for r in rev:
                j = len(srt)
                while j > 0 and srt[j - 1] > r:
                    ce += _mss(srt[j - 1], r)
                    j -= 1
                srt.insert(j, r)
            qe = qe - ce
            for r in run:
                den[r] += 1
            prepared.append((qe, el, tuple(den)))
        W = tuple(max(p[2][r] for p in prepared) if prepared else 0
                  for r in range(N))
        total = None
        for qe, el, den in prepared:
            u = tuple(W[r] - den[r] for r in range(N))
            cdu = sum(den[r] * u[t] * _mss(r, t)
                      for t in range(N) for r in range(t + 1, N))
            ext = el
            for r in range(N):
                for _ in range(u[r]):
                    ext = ext * self.S[r]
            ext = ext * Q(Fraction(qe) - cdu)
            total = ext if total is None else total + ext
        return total, W

    def _extend(self, el, w, Wc):
        u = tuple(Wc[r] - w[r] for r in range(N))
        cdu = sum(w[r] * u[t] * _mss(r, t)
                  for t in range(N) for r in range(t + 1, N))
        ext = el
        for r in range(N):
            for _ in range(u[r]):
                ext = ext * self.S[r]
        return ext * Q(Fraction(-cdu))

    @staticmethod
    def _mono_ratio_hl(a: HalfLaurent, b: HalfLaurent):
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

    def _to_hl(self, c: LaurentPoly) -> HalfLaurent:
        out = {}
        for k, v in c._coeffs.items():
            out[Fraction(self.ORI * k)] = out.get(Fraction(self.ORI * k), 0) + v
        return HalfLaurent(out)

    # ---- the dictionary: generator words --------------------------------

    @staticmethod
    def _qexp1(poly):
        items = list(poly._coeffs.items())
        assert len(items) == 1 and items[0][1] == 1, poly
        return items[0][0]

    def _gen_word_tokens(self, g):
        """The dressed-chord tokens for cone mult-gen g (= (1,i) short
        or (2,i) long), under the current dictionary offset a3."""
        a, i = g
        if a == 1:
            return _ts(i)
        return _tl(_f7(i - self._a3))

    def _forward_terms(self, lbl):
        """(lq-exponent, tokens) of the canonical label's skein image:
        the cone-ordered product of generator words, with the phase
        prod gens = q_int^{c} Canon[lbl] read off the A1A2kKAlg engine
        (so forward = lq^{-ORI c} . prod gen-words)."""
        lbl = tuple(lbl)
        if lbl == self._intr.identity():
            return (Fraction(0), ())
        gens, powers = self._cd.to_cone_label(lbl)
        order = self._cd.canonical_cone_order(gens)
        A = self._intr
        acc = None
        toks = []
        for g in order:
            glbl = self._cd.from_cone_label(frozenset({g}), {g: 1})
            gt = self._gen_word_tokens(g)
            for _ in range(powers.get(g, 0)):
                if acc is None:
                    acc = (glbl, Fraction(0))
                else:
                    prod = A.multiply(acc[0], glbl)
                    terms = dict(prod.terms)
                    assert len(terms) == 1, (lbl, g, terms)
                    nlbl = next(iter(terms))
                    acc = (nlbl, acc[1] + self._qexp1(terms[nlbl]))
                toks.extend(gt)
        assert acc is not None and tuple(acc[0]) == lbl, (lbl, acc)
        return (Fraction(-self.ORI * acc[1]), tuple(toks))

    def forward(self, lbl):
        key = tuple(lbl)
        if key not in self._fwd_cache:
            self._fwd_cache[key] = self._normalize(
                [self._forward_terms(key)])
        return self._fwd_cache[key]

    def _peel(self, a, b):
        a = tuple(a)
        b = tuple(b)
        if (a, b) in self._peel_cache:
            return self._peel_cache[(a, b)]
        ea, ta = self._forward_terms(a)
        eb, tb = self._forward_terms(b)
        num, W = self._normalize([(ea + eb, ta + tb)])
        hint_el = self._intr.multiply(a, b)
        hint = {tuple(l): c for l, c in hint_el.terms.items()}
        cands = {k: self.forward(k) for k in hint}
        Wc = tuple(max([W[r]] + [w[r] for (_e, w) in cands.values()])
                   for r in range(N))
        rem = self._extend(num, W, Wc)
        pool = {k: self._extend(n2, w2, Wc) for k, (n2, w2) in cands.items()}
        skein_co = {}
        while rem.terms:
            hit = None
            for lbl in sorted(rem.terms, key=str):
                owners = [k for k, c in pool.items() if lbl in c.terms]
                if len(owners) == 1:
                    hit = (lbl, owners[0])
                    break
            assert hit is not None, f"peel stuck for {a} * {b}"
            lbl, k = hit
            r = self._mono_ratio_hl(rem.terms[lbl], pool[k].terms[lbl])
            assert r is not None, f"non-monomial peel at {k}"
            e, sg = r
            skein_co[k] = (e, sg)
            rem = rem - pool[k] * Q(e) * sg
            pool.pop(k)
        assert not pool, (
            f"skein/intrinsic label-set mismatch at {a} * {b}: "
            f"intrinsic labels {sorted(pool)} absent from the engine "
            f"product")
        self._peel_cache[(a, b)] = (hint, skein_co)
        return hint, skein_co

    def _generators(self):
        return ([self._intr.identity()]
                + [((1, i, 1),) for i in range(N)]
                + [((2, i, 1),) for i in range(N)])

    def _try_calibrate(self):
        """Attempt the δ-calibration at the current a3.  Returns the δ
        dict on success, or None on any peel/consistency failure."""
        self._fwd_cache.clear()
        self._peel_cache.clear()
        self._mult_cache.clear()
        ident = self._intr.identity()
        gens = self._generators()
        eqs = []
        try:
            for g1 in gens:
                for g2 in gens:
                    hint, sc = self._peel(g1, g2)
                    for k, (e, sg) in sc.items():
                        cintr = hint[k]
                        rel = self._mono_ratio_hl(Q(e) * sg,
                                                  self._to_hl(cintr))
                        if rel is None or rel[1] != 1:
                            return None
                        xshift = rel[0]
                        eq = {}
                        eq[g1] = eq.get(g1, 0) + 1
                        eq[g2] = eq.get(g2, 0) + 1
                        eq[k] = eq.get(k, 0) - 1
                        eqs.append((eq, xshift))
        except AssertionError:
            return None
        eqs.append(({ident: 1}, Fraction(0)))
        labels = sorted({l for eq, _ in eqs for l in eq}, key=str)
        idx = {l: i for i, l in enumerate(labels)}
        rows = [[Fraction(0)] * len(labels) for _ in eqs]
        rhs = []
        for r, (eq, val) in enumerate(eqs):
            for l, co in eq.items():
                rows[r][idx[l]] = Fraction(co)
            rhs.append(Fraction(val))
        consistent, sol = _solve_exact(rows, rhs)
        if not consistent:
            return None
        return {l: sol[idx[l]] for l in labels}

    def _resolve(self):
        """Calibrate δ at the fixed dictionary (a3 = 0).  The
        δ-system being consistent — every generator product's label
        set matching the intrinsic and every product over-determining
        the same per-label δ — IS the certification that the dressed
        chords present exactly the A₄ heptagon."""
        if self._delta is not None:
            return
        delta = self._try_calibrate()
        if delta is None:
            raise ValueError(
                "SkeinHeptagonKAlg: the skein↔intrinsic calibration is "
                "inconsistent at a3=0 (the dressing or family "
                "assignment would be wrong)")
        self._delta = delta
        self._fwd_cache.clear()
        self._peel_cache.clear()
        self._mult_cache.clear()

    def _delta_of(self, lbl):
        lbl = tuple(lbl)
        if lbl == self._intr.identity():
            return Fraction(0)
        self._resolve()
        if lbl not in self._delta:
            raise KeyError(
                f"δ of {lbl} not calibrated (multiplicands must be "
                f"generators or generator-products)")
        return self._delta[lbl]

    # ---- KAlgebra primitives --------------------------------------------

    def coefficient_ring(self):
        return self._intr.coefficient_ring()

    def identity(self):
        return self._intr.identity()

    def multiply(self, a, b) -> Element:
        a = tuple(a)
        b = tuple(b)
        key = (a, b)
        if key in self._mult_cache:
            return self._mult_cache[key]
        self._resolve()
        da = self._delta_of(a)
        db = self._delta_of(b)
        hint, skein_co = self._peel(a, b)
        out = {}
        for k, (e, sg) in skein_co.items():
            cintr = hint[k]
            rel = self._mono_ratio_hl(Q(e) * sg, self._to_hl(cintr))
            assert rel is not None and rel[1] == 1, (
                f"skein/intrinsic coefficient-sign mismatch at "
                f"{a} * {b}, label {k}")
            xshift = rel[0]
            want = da + db - xshift
            if k in self._delta:
                assert self._delta[k] == want, (
                    f"δ inconsistency at {k} in {a}*{b}: "
                    f"{self._delta[k]} vs {want}")
            else:
                self._delta[k] = want
            out[k] = cintr
        el = Element(out)
        self._mult_cache[key] = el
        return el

    def rho(self, a):
        r = self._intr.rho(tuple(a))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return r

    def rho_inverse(self, a):
        r = self._intr.rho_inverse(tuple(a))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return r

    def trace(self, a, K: int = 20):
        return self._intr.trace(tuple(a), K)

    def _label_section_decompose(self, label):
        """The trivial section coordinate, matching `r_label_decompose`
        (flavour, if any, lives in the intrinsic coefficient ring; the
        canonical basis is flavour-neutral); `to_R_form` routes through it."""
        return (tuple(label), self.coefficient_ring().one())

    def r_label_decompose(self, label):
        """Trivial flavour-lift coordinate `(label, χ₀)`: the heptagon skein
        chart carries flavour in its coefficient ring, so the canonical basis
        is flavour-neutral.  Implemented directly (independent of
        `_label_section_decompose`; `forget()` / ring-hom promotion read
        this)."""
        return tuple(label), self.coefficient_ring().one_basis()

    def r_label_compose(self, section, r_basis_label):
        return tuple(section)

    # ---- the iso witness --------------------------------------------------

    def build_iso(self):
        from kalgebra_iso import KAlgebraIso
        one = LaurentPoly.one()

        def _id(lbl):
            return Element({tuple(lbl): one})

        return KAlgebraIso(self, self._intr, _id, _id,
                           name="heptagon[skein→a1a2k]")
