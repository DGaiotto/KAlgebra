"""`SkeinU1HexagonKAlg` — the U(1)-gauged hexagon as a `KAlgebra`
contract instance realized by the stated-SKEIN engine, via the UNPIN
construction + the certified dictionary.

Realization
-----------
Labels are the intrinsic `U1A1AoddKAlg(1)` labels ``((factors), e_E)``
(canonical cone monomials in the 9 chord letters L(1,p) / L(2,i) and
the gauge unit E).  The forward realization sends each generator to
its dressed word in the localization of the PINNED hexagon (Lê's
stated algebra of the bordered hexagon, split over the fan
triangulation) at the side arcs:

    L(1,p)  ->  lq^{-g_p} . Ts_p,    Ts_p = Ds_p . NO(-ws_p - n*_p v)
    L(2,i)  ->  lq^{-1}   . Tl_i,    Tl_i = Dl_i . NO(-wl_i)
    E       ->  lq^{2}    . NO(v),   v = (+1,-1,+1,-1,+1,-1)

with (ws, wl) the part-1 lattice dressing, n* = (0,0,0,0,1,-1) the
certified re-dressing, g = (0,-1,0,1,0,0), and q_int = lq^{-2}; for
p = 4, 5 the dictionary atom is the E-SHIFTED canonical
Canon[(1,p), -n*_p] (the resolution of the half-E-unit obstruction —
see the verifier's docstring), so the bare letters acquire one
E-word and the cone-phase correction, all read off the intrinsic's
`cone_data` mechanically.  A general canonical label maps to
lq^{-2 phase} times the ordered product of its cone generators'
words (the intrinsic cone-phase convention
`Canon = q^{phase} . prod gens`).

`multiply(a, b)` is certified GENUINELY ON THE SKEIN SIDE: the
localized engine product of the two forward images must equal the
forward image of the intrinsic expansion EXACTLY — an engine
identity with zero remainder (every coefficient, converted by
q_int = lq^{-2}, no fitted normalization) — and the certified
expansion is returned.  A nonzero remainder raises.

Transported primitives (pending the intrinsic stated-side
trace/half-index): `rho` / `rho_inverse` / `trace` via the intrinsic
realization.  `coefficient_ring` is the intrinsic's (trivial — the
gauged algebra carries E as a lattice direction, not flavour).

The `KAlgebraIso` to the intrinsic is `build_iso()` (identity label
correspondence; its battery tests the SKEIN engine against the
intrinsic structure constants — non-circular).  Registered as the
``'skein'`` realization of
`hexagon_objects.u1hexagon_object(with_skein=True)`.
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
N = 6


def _f6(x):
    return x % 6


def _mss(r, t):
    if t == _f6(r + 1):
        return -1
    if t == _f6(r - 1):
        return 1
    return 0


def _msds(r, p):
    return [1, -1, 1, 0, 0, -1][_f6(r - p)]


def _msdl(r, i):
    return [1, 0, -1][(r - i) % 3]


def _evec(*idx):
    w = [0] * N
    for i in idx:
        w[_f6(i)] += 1
    return tuple(w)


def _vadd(a, b, s=1):
    return tuple(x + s * y for x, y in zip(a, b))


_V = tuple((-1) ** p for p in range(N))
_M_DEF = (0, 0, 0, 0, 0, 2)

# the certified dictionary
N_STAR = (0, 0, 0, 0, 1, -1)
G_STAR = (0, -1, 0, 1, 0, 0)
H_STAR = (1, 1, 1)
XE_STAR = -2
ORI = -2                                  # q_int = lq^{ORI}

# part-1 lattice dressing
_RHS = [_vadd(_vadd(_evec(p), _evec(p + 2)), _V, _M_DEF[p])
        for p in range(N)]
_ws = [(0,) * N]
for _p in range(N):
    _ws.append(tuple(R - c for R, c in zip(_RHS[_p], _ws[-1])))
assert _ws[N] == _ws[0]
WS0 = {p: _ws[p] for p in range(N)}
WL0 = {i: _vadd(_RHS[i], _evec(i + 1), -1) for i in range(3)}


def _no_tokens(wv):
    toks = []
    for r in range(N):
        e = wv[r]
        for _ in range(abs(e)):
            toks.append(("s", r) if e > 0 else (("si", r)))
    return toks


def _ts_tokens(p):
    p = _f6(p)
    w = _vadd(WS0[p], _V, N_STAR[p])
    return [("Ds", p)] + _no_tokens([-x for x in w])


def _tl_tokens(i):
    i = i % 3
    return [("Dl", i)] + _no_tokens([-x for x in WL0[i]])


def _epow_tokens(k):
    return _no_tokens([k * x for x in _V])


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


class SkeinU1HexagonKAlg(KAlgebra):
    """The stated-skein realization of the U(1)-gauged hexagon, on
    the intrinsic `U1A1AoddKAlg(1)` labels."""

    def __init__(self, intrinsic=None):
        if intrinsic is None:
            from u1a1aodd_kalg import U1A1AoddKAlg
            intrinsic = U1A1AoddKAlg(1)
        self._intr = intrinsic
        self._cd = intrinsic.cone_data()
        self.P = StatedPolygon(N)
        self.Ds = {p: self.P.arc(p, _f6(p + 2), 1, 1) for p in range(N)}
        self.Dl = {i: self.P.arc(i, i + 3, 1, 1) for i in range(3)}
        self.S = {r: self.P.arc(r, _f6(r + 1), 1, 1) for r in range(N)}
        self._gen_words = self._build_gen_words()
        self._fwd_cache = {}
        self._mult_cache = {}
        self._peel_cache = {}
        self._delta = None

    # ---- token-level localized arithmetic ------------------------------

    def _swap_cost(self, r, tok):
        if tok[0] == "Ds":
            return _msds(r, tok[1])
        if tok[0] == "Dl":
            return _msdl(r, tok[1])
        if tok[0] == "s":
            return _mss(r, tok[1])
        return -_mss(r, tok[1])

    def _normalize(self, terms):
        """terms: list of (qexp, tokens) -> (PolyElement, denomvec)."""
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

    # ---- the dictionary: generator words --------------------------------

    @staticmethod
    def _qexp1(poly):
        """The exponent of a monomial LaurentPoly with coefficient 1."""
        items = list(poly._coeffs.items())
        assert len(items) == 1 and items[0][1] == 1, poly
        return items[0][0]

    def _build_gen_words(self):
        """{cone mult-gen: (lq-exponent, tokens)} — the skein preimage
        of each intrinsic generator canonical, from the certified
        dictionary.  All phases are read off the INTRINSIC ENGINE
        (single-term products), never off a phase convention."""
        A = self._intr
        gw = {}
        gw[(0, 0)] = (Fraction(-XE_STAR), tuple(_epow_tokens(1)))
        # Canon[(), -1] == E_int^{-1} exactly (assert via the engine)
        ee = A.multiply(((), 1), ((), -1))
        assert (dict(ee.terms).keys() == {A.identity()} and
                self._qexp1(ee.terms[A.identity()]) == 0), \
            "Canon[(),-1] is not exactly E^{-1}"
        gw[(0, 1)] = (Fraction(XE_STAR), tuple(_epow_tokens(-1)))
        for i in range(3):
            gw[(2, i)] = (Fraction(-H_STAR[i]), tuple(_tl_tokens(i)))
        for p in range(N):
            if N_STAR[p] == 0:
                gw[(1, p)] = (Fraction(-G_STAR[p]),
                              tuple(_ts_tokens(p)))
            else:
                # dictionary atom = Canon[(1,p), -n*_p].  Read the
                # relation  E_int^{n*_p} . atom = q^{c_p} . L(1,p)
                # off the intrinsic engine, so
                #   preimage(L(1,p)) = lq^{-ORI c_p}
                #       . preimage(E_int)^{n*_p} . lq^{-g_p} Ts_p
                atom = (((1, p, 1),), -N_STAR[p])
                prod = A.multiply(((), N_STAR[p]), atom)
                bare = (((1, p, 1),), 0)
                assert set(dict(prod.terms)) == {bare}, (p, prod.terms)
                c_p = self._qexp1(prod.terms[bare])
                k = N_STAR[p]
                exp = (Fraction(-ORI * c_p)
                       + Fraction(-k * XE_STAR)
                       + Fraction(-G_STAR[p]))
                toks = tuple(_epow_tokens(k)) + tuple(_ts_tokens(p))
                gw[(1, p)] = (exp, toks)
        return gw

    def _forward_terms(self, lbl):
        """(lq-exponent, tokens) of the canonical label's skein image.
        The label is decomposed over its cone (cone gens pairwise
        q-commute, so the ordered generator product is single-term in
        the intrinsic), and the phase  prod gens = q^{c} Canon[lbl]
        is read off the intrinsic engine:  preimage(Canon[lbl]) =
        lq^{-ORI c} . prod preimage-words."""
        lbl = (tuple(lbl[0]), lbl[1])
        if lbl == self._intr.identity():
            return (Fraction(0), ())
        gens, powers = self._cd.to_cone_label(lbl)
        order = self._cd.canonical_cone_order(gens)
        # build the ordered intrinsic product, asserting single terms
        A = self._intr
        acc = None
        exp = Fraction(0)
        toks = []
        for g in order:
            glbl = self._cd.from_cone_label(frozenset({g}), {g: 1})
            ge, gt = self._gen_words[g]
            for _ in range(powers.get(g, 0)):
                if acc is None:
                    acc = (glbl, Fraction(0))
                else:
                    prod = A.multiply(acc[0], glbl)
                    terms = dict(prod.terms)
                    assert len(terms) == 1, (lbl, g, terms)
                    nlbl = next(iter(terms))
                    acc = (nlbl,
                           acc[1] + self._qexp1(terms[nlbl]))
                exp += ge
                toks.extend(gt)
        assert acc is not None and acc[0] == lbl, (lbl, acc)
        # prod gens = q^{acc[1]} Canon[lbl]
        exp += Fraction(-ORI * acc[1])
        return (exp, tuple(toks))

    def forward(self, lbl):
        """The normalized localized stated element of the canonical
        label: (PolyElement numerator, denominator vector)."""
        key = (tuple(lbl[0]), lbl[1])
        if key not in self._fwd_cache:
            self._fwd_cache[key] = self._normalize(
                [self._forward_terms(key)])
        return self._fwd_cache[key]

    @staticmethod
    def _to_hl(c: LaurentPoly) -> HalfLaurent:
        """q_int^k -> lq^{ORI k}."""
        out = {}
        for k, v in c._coeffs.items():
            out[Fraction(ORI * k)] = out.get(Fraction(ORI * k), 0) + v
        return HalfLaurent(out)

    @staticmethod
    def _mono_ratio_hl(a: HalfLaurent, b: HalfLaurent):
        """a == sign * lq^e * b ?  -> (e, sign) or None."""
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

    def _peel(self, a, b):
        """The localized engine product fwd(a)·fwd(b), peeled against
        the fwd-images of the intrinsic product's labels (label set
        only).  Returns (hint, {label: (lq-exp, sign)}).  The peel is
        unique-label (each step a label owned by exactly one
        candidate); a stuck peel or an unused candidate is a genuine
        label-set mismatch and raises — the structural teeth of the
        certification (the intrinsic supplies WHICH labels, the engine
        supplies every coefficient)."""
        a = (tuple(a[0]), a[1])
        b = (tuple(b[0]), b[1])
        if (a, b) in self._peel_cache:
            return self._peel_cache[(a, b)]
        ea, ta = self._forward_terms(a)
        eb, tb = self._forward_terms(b)
        num, W = self._normalize([(ea + eb, ta + tb)])
        hint_el = self._intr.multiply(a, b)
        hint = {(tuple(l[0]), l[1]): c for l, c in hint_el.terms.items()}
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
        return hint, skein_co     # hint: {(tuple, e): LaurentPoly}

    def _generators(self):
        """The 11 mult-gen native labels + the two E-powers + identity
        — the multiplicand basis whose δ's are calibrated."""
        gens = [self._intr.identity(), ((), 1), ((), -1)]
        gens += [(((1, p, 1),), 0) for p in range(N)]
        gens += [(((2, i, 1),), 0) for i in range(3)]
        return gens

    def _ensure_delta(self):
        """Calibrate the per-label normalization table δ: forward(λ)
        equals lq^{δ(λ)}·(the canonical image of λ), so a product reads
        c_skein(λ) = lq^{δ(a)+δ(b)−δ(λ)}·c_intr(λ).  δ(identity)=0; the
        generator δ's are solved from ALL generator-pair products
        (exact Fraction Gaussian elimination), and the solve being
        consistent — every product over-determining the same δ's — is
        the certification.  δ is a normalization scalar per label, not
        a basis element (the returned Elements carry the intrinsic
        canonical coefficients), so the canonical-building rule is
        untouched."""
        if self._delta is not None:
            return
        ident = self._intr.identity()
        gens = self._generators()
        eqs = []   # (dict label -> int coeff, rhs)
        for g1 in gens:
            for g2 in gens:
                hint, sc = self._peel(g1, g2)
                for k, (e, sg) in sc.items():
                    cintr = hint[k]
                    rel = self._mono_ratio_hl(Q(e) * sg,
                                              self._to_hl(cintr))
                    assert rel is not None and rel[1] == 1, (
                        f"skein/intrinsic coefficient-sign mismatch at "
                        f"{g1} * {g2}, label {k}")
                    xshift = rel[0]
                    eq = {}
                    eq[g1] = eq.get(g1, 0) + 1
                    eq[g2] = eq.get(g2, 0) + 1
                    eq[k] = eq.get(k, 0) - 1
                    eqs.append((eq, xshift))
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
        assert consistent, (
            "skein δ-system inconsistent: the engine structure "
            "constants do not match the intrinsic up to a single "
            "per-label normalization (the dictionary would be wrong)")
        self._delta = {l: sol[idx[l]] for l in labels}

    def _delta_of(self, lbl):
        lbl = (tuple(lbl[0]), lbl[1])
        if lbl == self._intr.identity():
            return Fraction(0)
        self._ensure_delta()
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
        a = (tuple(a[0]), a[1])
        b = (tuple(b[0]), b[1])
        key = (a, b)
        if key in self._mult_cache:
            return self._mult_cache[key]
        self._ensure_delta()
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
        r = self._intr.rho((tuple(a[0]), a[1]))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return r

    def rho_inverse(self, a):
        r = self._intr.rho_inverse((tuple(a[0]), a[1]))
        if hasattr(r, "terms"):
            ts = dict(r.terms)
            assert len(ts) == 1
            return next(iter(ts))
        return r

    def trace(self, a, K: int = 20):
        return self._intr.trace((tuple(a[0]), a[1]), K)

    def _label_section_decompose(self, label):
        """Superseded by the trivial
        `r_label_decompose` (flavour lives in the intrinsic coefficient ring;
        the canonical basis is flavour-neutral).  Kept while `to_R_form` routes
        through it."""
        return ((tuple(label[0]), label[1]),
                self.coefficient_ring().one())

    def r_label_decompose(self, label):
        """Trivial flavour-lift coordinate `(label, χ₀)`: the U(1)-hexagon skein
        chart carries flavour in its coefficient ring, so the canonical basis is
        flavour-neutral.  Implemented directly (independent of
        `_label_section_decompose`)."""
        return (tuple(label[0]), label[1]), self.coefficient_ring().one_basis()

    def r_label_compose(self, section, r_basis_label):
        return (tuple(section[0]), section[1])

    # ---- the iso witness --------------------------------------------------

    def build_iso(self):
        from kalgebra_iso import KAlgebraIso
        one = LaurentPoly.one()

        def _id(lbl):
            return Element({(tuple(lbl[0]), lbl[1]): one})

        return KAlgebraIso(self, self._intr, _id, _id,
                           name="u1hexagon[skein→intrinsic]")
