"""`weyl_torus_ring` — the 𝖖-rational function ring of the maximal torus,
parameterized by a `RootDatum` (Layer 1 of the `AbeKAlgebra` substrate redesign).

The ring is the localized group algebra of the weight lattice

    R(datum) = S[P] [ D^{-1} ],   D = ⟨ 1 − 𝖖^k v^α : α ∈ Φ, k ∈ Z ⟩

with monomials `v^λ` (λ ∈ P) and denominators built from the uniform factor
`1 − 𝖖^k v^α`, α a root.  The **scalar coefficient ring `S`** is either:

  * `Z[𝖖^±]` (`laurent_poly.LaurentPoly`) — the default (`scalar=None`); or
  * `R[𝖖^±]` (`zplus_ring.RLaurent`) for a Z₊-ring `R` (`scalar=R`) — so the
    **flavour fugacities live in the coefficient ring**, exactly as the KAlgebra
    contract intends (`coefficient_ring = R(G_f)`), with the gauge weight lattice
    and root denominators unchanged.

`TorusLaurent` carries its scalar; `TorusRational` infers it from the numerator.
The Z-path (`scalar=None`) keeps the raw-int fast multiply and is behaviourally
unchanged; the R-path uses the coefficients' own `+`/`*`/`bar`.
"""
from __future__ import annotations

from laurent_poly import LaurentPoly
from root_datum import RootDatum
from zplus_ring import RLaurent

Weight = tuple
DenKey = tuple        # (alpha_pos: Weight, k: int)


# ---------------------------------------------------------------------------
# scalar-coefficient helpers — dispatch on `scalar` (None = Z[𝖖^±]/LaurentPoly,
# else a ZPlusRing R, coefficients = RLaurent = R[𝖖^±])
# ---------------------------------------------------------------------------
def _s_zero(scalar):
    return LaurentPoly.zero() if scalar is None else RLaurent.zero(scalar)


def _s_const(scalar, c):
    """`c · 𝖖^0` (c an int or, for R, an int / RElement)."""
    return LaurentPoly({0: c}) if scalar is None else RLaurent(scalar, {0: c})


def _s_qmono(scalar, k, c=1):
    """`c · 𝖖^k`."""
    return LaurentPoly({k: c}) if scalar is None else RLaurent(scalar, {k: c})


def _s_qshift(coeff, k, scalar):
    """`coeff · 𝖖^k`."""
    if k == 0:
        return coeff
    if scalar is None:
        return LaurentPoly({e + k: c for e, c in coeff._coeffs.items()})
    return RLaurent(scalar, {e + k: c for e, c in coeff.coeffs.items()})


def _s_bar(coeff, scalar):
    """`𝖖 ↦ 𝖖⁻¹` on a scalar (R-side untouched)."""
    if scalar is None:
        return LaurentPoly({-e: c for e, c in coeff._coeffs.items()})
    return coeff.bar()


def _s_is_lp(scalar, o):
    """Is `o` a bare scalar of the configured coefficient ring?"""
    return isinstance(o, LaurentPoly) if scalar is None else isinstance(o, RLaurent)


# ===========================================================================
# TorusLaurent : S[P]   (S = Z[𝖖^±] or R[𝖖^±])
# ===========================================================================
class TorusLaurent:
    """Laurent polynomial `Σ_λ c_λ(𝖖) · v^λ`, `c_λ ∈ S`, λ ∈ P ⊂ Z^d.

    Stored as `dict[Weight, S-coeff]` (zeros dropped).  `scalar` selects S."""

    __slots__ = ("datum", "_d", "_t", "_scalar")

    def __init__(self, datum: RootDatum, terms: dict, scalar=None):
        self.datum = datum
        self._d = datum.dim
        self._scalar = scalar
        t = {}
        for e, lp in terms.items():
            if lp is not None and not lp.is_zero():
                t[tuple(e)] = lp
        self._t = t

    # ----- constructors -----
    @classmethod
    def zero(cls, datum, scalar=None):
        return cls(datum, {}, scalar)

    @classmethod
    def one(cls, datum, scalar=None):
        return cls(datum, {(0,) * datum.dim: _s_const(scalar, 1)}, scalar)

    @classmethod
    def monomial(cls, datum, weight, lp=None, scalar=None):
        if lp is None:
            lp = _s_const(scalar, 1)
        return cls(datum, {tuple(weight): lp}, scalar)

    @classmethod
    def from_scalar(cls, datum, s, scalar=None):
        lp = s if _s_is_lp(scalar, s) else _s_const(scalar, int(s))
        return cls(datum, {(0,) * datum.dim: lp}, scalar)

    # ----- accessors -----
    @property
    def terms(self): return dict(self._t)

    @property
    def scalar(self): return self._scalar

    def is_zero(self): return not self._t

    def coeff(self, weight):
        return self._t.get(tuple(weight), _s_zero(self._scalar))

    # ----- arithmetic -----
    def _coerce(self, o):
        if isinstance(o, TorusLaurent):
            return o
        if isinstance(o, int) or _s_is_lp(self._scalar, o):
            return TorusLaurent.from_scalar(self.datum, o, self._scalar)
        return None

    def __add__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        out = dict(self._t)
        for e, lp in o._t.items():
            s = (out[e] + lp) if e in out else lp
            if s.is_zero():
                out.pop(e, None)
            else:
                out[e] = s
        return TorusLaurent(self.datum, out, self._scalar)

    __radd__ = __add__

    def __neg__(self):
        return TorusLaurent(self.datum, {e: -lp for e, lp in self._t.items()},
                            self._scalar)

    def __sub__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else self + (-o)

    def __mul__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        sc = self._scalar
        if self.is_zero() or o.is_zero():
            return TorusLaurent.zero(self.datum, sc)
        d = self._d
        out = {}
        if sc is None:
            # raw-int fast path (Z[𝖖^±]) — behaviourally unchanged
            out_raw: dict = {}
            for e1, c1 in self._t.items():
                for e2, c2 in o._t.items():
                    e = tuple(e1[i] + e2[i] for i in range(d))
                    acc = out_raw.setdefault(e, {})
                    for q1, k1 in c1._coeffs.items():
                        for q2, k2 in c2._coeffs.items():
                            qe = q1 + q2
                            nv = acc.get(qe, 0) + k1 * k2
                            if nv:
                                acc[qe] = nv
                            else:
                                acc.pop(qe, None)
            out = {e: LaurentPoly(d2) for e, d2 in out_raw.items() if d2}
        else:
            for e1, c1 in self._t.items():
                for e2, c2 in o._t.items():
                    e = tuple(e1[i] + e2[i] for i in range(d))
                    prod = c1 * c2
                    out[e] = (out[e] + prod) if e in out else prod
            out = {e: c for e, c in out.items() if not c.is_zero()}
        return TorusLaurent(self.datum, out, sc)

    __rmul__ = __mul__

    def __pow__(self, k: int):
        if k < 0:
            raise ValueError("TorusLaurent: negative powers need inversion")
        out = TorusLaurent.one(self.datum, self._scalar)
        base = self
        while k > 0:
            if k & 1:
                out = out * base
            k >>= 1
            if k:
                base = base * base
        return out

    def q_shift(self, c):
        """`v^λ ↦ 𝖖^{⟨c,λ⟩} v^λ` (the normal-ordering shift), via datum pairing."""
        dat, sc = self.datum, self._scalar
        out = {}
        for e, lp in self._t.items():
            s = dat.shift_pairing(c, e)
            out[e] = lp if s == 0 else _s_qshift(lp, s, sc)
        return TorusLaurent(dat, out, sc)

    def weyl_act(self, w):
        """`w · (Σ c_λ v^λ) = Σ c_λ v^{wλ}`."""
        dat = self.datum
        out = {}
        for e, lp in self._t.items():
            we = dat.act(w, e)
            out[we] = (out[we] + lp) if we in out else lp
        return TorusLaurent(dat, {e: lp for e, lp in out.items() if not lp.is_zero()},
                            self._scalar)

    def bar(self):
        """𝖖 ↦ 𝖖⁻¹ with v fixed (R-side untouched)."""
        return TorusLaurent(self.datum,
                            {e: _s_bar(lp, self._scalar) for e, lp in self._t.items()},
                            self._scalar)

    def vinv(self):
        """v ↦ 1/v (𝖖 fixed): negate every weight."""
        return TorusLaurent(self.datum,
                            {tuple(-x for x in e): lp for e, lp in self._t.items()},
                            self._scalar)

    def __eq__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        return self._t == o._t

    def __repr__(self):
        if not self._t:
            return "0"
        parts = []
        for e, lp in sorted(self._t.items()):
            ve = "*".join(f"v{i}^{e[i]}" for i in range(self._d) if e[i]) or "1"
            parts.append(f"({lp})*{ve}")
        return " + ".join(parts)


# ===========================================================================
# division by a single factor (1 − 𝖖^k v^β), β a root
# ===========================================================================
def _divide_by_factor(terms: dict, beta: Weight, k: int, d: int, scalar=None):
    """Exact division `(Σ c_λ v^λ) / (1 − 𝖖^k v^β)`; returns the quotient dict, or
    `None` if the factor does not divide exactly.

    Per β-line `{base + nβ}` the equation `P = (1−t)Q`, `t = 𝖖^k v^β`, is the
    first-order recurrence `q_n = p_n + 𝖖^k·q_{n−1}`; the factor divides iff each
    line's top overflow vanishes (≡ the substitution `v^β = 𝖖^{−k}` kills P)."""
    if not terms:
        return {}
    zero = _s_zero(scalar)
    D = sum(b * b for b in beta)                    # φ(β) = β·β > 0 (β ≠ 0)
    lines: dict = {}
    for lam, lp in terms.items():
        g = sum(lam[t] * beta[t] for t in range(d))
        key = tuple(lam[t] * D - g * beta[t] for t in range(d))
        lines.setdefault(key, {})[g] = (lam, lp)
    quo: dict = {}
    for gmap in lines.values():
        gmin, gmax = min(gmap), max(gmap)
        base_w = gmap[gmin][0]
        prev = zero                                  # q_{n-1}
        g = gmin
        while g <= gmax:
            p_g = gmap[g][1] if g in gmap else zero
            qg = p_g + _s_qshift(prev, k, scalar)    # q_n = p_n + 𝖖^k q_{n-1}
            if g == gmax:
                if not qg.is_zero():                 # nonzero overflow ⇒ not divisible
                    return None
            elif not qg.is_zero():
                wt = tuple(base_w[t] + ((g - gmin) // D) * beta[t] for t in range(d))
                quo[wt] = qg
            prev = qg
            g += D
    return quo


# ===========================================================================
# TorusRational : R(datum) = S[P][D^{-1}]
# ===========================================================================
class TorusRational:
    """A 𝖖-rational function `num / ∏ (1 − 𝖖^k v^α)^mult`, α ∈ Φ⁺.  `num` is a
    `TorusLaurent`; the scalar ring is inferred from `num`."""

    __slots__ = ("datum", "_d", "_num", "_den", "_simp")

    def __init__(self, datum, num: TorusLaurent, den: dict | None = None):
        self.datum = datum
        self._d = datum.dim
        self._simp = None
        sc = num._scalar
        d2 = {}
        if den:
            for (a, k), m in den.items():
                if not m:
                    continue
                a, k, m = tuple(a), int(k), int(m)
                a_pos, flipped = datum.orient(a)
                if flipped:
                    mono = TorusLaurent.monomial(datum, a_pos, _s_qmono(sc, -k, -1), sc)
                    num = num * (mono ** m)
                    key = (a_pos, -k)
                else:
                    key = (a_pos, k)
                d2[key] = d2.get(key, 0) + m
            d2 = {key: m for key, m in d2.items() if m != 0}
        self._num = num
        self._den = d2

    @property
    def _sc(self): return self._num._scalar

    # ----- constructors -----
    @classmethod
    def zero(cls, datum, scalar=None):
        return cls(datum, TorusLaurent.zero(datum, scalar))

    @classmethod
    def one(cls, datum, scalar=None):
        return cls(datum, TorusLaurent.one(datum, scalar))

    @classmethod
    def from_laurent(cls, x: TorusLaurent):
        return cls(x.datum, x)

    @classmethod
    def from_scalar(cls, datum, s, scalar=None):
        return cls(datum, TorusLaurent.from_scalar(datum, s, scalar))

    @classmethod
    def factor_inv(cls, datum, alpha, k, scalar=None):
        """`1 / (1 − 𝖖^k v^α)` for any root α; `__init__` canonicalizes α to Φ⁺."""
        return cls(datum, TorusLaurent.one(datum, scalar), {(tuple(alpha), int(k)): 1})

    @classmethod
    def factor(cls, datum, alpha, k, scalar=None):
        """The Laurent factor `1 − 𝖖^k v^α` itself (α need not be positive)."""
        return cls(datum, _factor_laurent(datum, alpha, k, scalar))

    # ----- accessors -----
    @property
    def num(self): return self._num

    @property
    def den(self): return dict(self._den)

    @property
    def scalar(self): return self._sc

    def is_zero(self): return self._num.is_zero()

    def den_as_laurent(self) -> TorusLaurent:
        out = TorusLaurent.one(self.datum, self._sc)
        for (a, k), m in self._den.items():
            out = out * (_factor_laurent(self.datum, a, k, self._sc) ** m)
        return out

    # ----- arithmetic -----
    def _coerce(self, o):
        if isinstance(o, TorusRational):
            return o
        if isinstance(o, TorusLaurent):
            return TorusRational.from_laurent(o)
        if isinstance(o, int) or _s_is_lp(self._sc, o):
            return TorusRational.from_scalar(self.datum, o, self._sc)
        return None

    def __neg__(self):
        return TorusRational(self.datum, -self._num, self._den)

    def __add__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        common, a_extra, b_extra = {}, {}, {}
        for key in set(self._den) | set(o._den):
            sa, sb = self._den.get(key, 0), o._den.get(key, 0)
            m = max(sa, sb)
            common[key] = m
            if m - sa:
                a_extra[key] = m - sa
            if m - sb:
                b_extra[key] = m - sb
        na = self._num * _den_laurent(self.datum, a_extra, self._sc)
        nb = o._num * _den_laurent(self.datum, b_extra, self._sc)
        return TorusRational(self.datum, na + nb, common)

    __radd__ = __add__

    def __sub__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is None else self + (-o)

    __rsub__ = lambda self, other: (self._coerce(other) - self)  # noqa: E731

    def __mul__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        den = dict(self._den)
        for key, m in o._den.items():
            den[key] = den.get(key, 0) + m
        return TorusRational(self.datum, self._num * o._num, den)

    __rmul__ = __mul__

    def __pow__(self, k: int):
        if k < 0:
            raise ValueError("TorusRational: negative powers need inversion")
        out = TorusRational.one(self.datum, self._sc)
        base = self
        while k > 0:
            if k & 1:
                out = out * base
            k >>= 1
            if k:
                base = base * base
        return out

    def q_shift(self, c):
        """`v^λ ↦ 𝖖^{⟨c,λ⟩} v^λ`.  Denominator: `(α,k) ↦ (α, k+⟨c,α⟩)` — no
        monomial extracted (the leading `1` of `1 − 𝖖^k v^α` is shift-invariant)."""
        dat = self.datum
        new_den = {}
        for (a, k), m in self._den.items():
            nk = k + dat.shift_pairing(c, a)
            new_den[(a, nk)] = new_den.get((a, nk), 0) + m
        return TorusRational(dat, self._num.q_shift(c), new_den)

    def simplify(self):
        """Cancel common `(1 − 𝖖^k v^α)` factors between numerator and denominator."""
        if self._simp is not None:
            return self._simp
        num = self._num
        den = dict(self._den)
        for (a, k), mult in list(den.items()):
            while mult > 0:
                q = _divide_by_factor(num._t, a, k, self._d, self._sc)
                if q is None:
                    break
                num = TorusLaurent(self.datum, q, self._sc)
                mult -= 1
            if mult <= 0:
                del den[(a, k)]
            else:
                den[(a, k)] = mult
        res = TorusRational(self.datum, num, den)
        res._simp = res
        self._simp = res
        return res

    def weyl_act(self, w):
        """`w·(num/den)`: act on num and on each denominator root (k unchanged);
        `__init__` re-canonicalizes the (possibly now-negative) image roots."""
        dat = self.datum
        new_den = {(dat.act(w, a), k): m for (a, k), m in self._den.items()}
        return TorusRational(dat, self._num.weyl_act(w), new_den)

    def bar(self):
        """𝖖 ↦ 𝖖⁻¹ with v fixed: `num ↦ bar(num)`, `(α,k) ↦ (α,−k)`."""
        new_den = {}
        for (a, k), m in self._den.items():
            new_den[(a, -k)] = new_den.get((a, -k), 0) + m
        return TorusRational(self.datum, self._num.bar(), new_den)

    def vinv(self):
        """v ↦ 1/v (𝖖 fixed): on `num` and on each denominator root (`α ↦ −α`)."""
        new_den = {(tuple(-x for x in a), k): m for (a, k), m in self._den.items()}
        return TorusRational(self.datum, self._num.vinv(), new_den)

    def __eq__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        return (self._num * o.den_as_laurent()) == (o._num * self.den_as_laurent())

    def __repr__(self):
        if self.is_zero():
            return "0"
        if not self._den:
            return repr(self._num)
        dparts = []
        for (a, k), m in sorted(self._den.items()):
            ve = "*".join(f"v{i}^{a[i]}" for i in range(self._d) if a[i]) or "1"
            base = f"(1 - q^{k}*{ve})"
            dparts.append(base if m == 1 else f"{base}^{m}")
        return f"({self._num}) / ({'*'.join(dparts)})"


# ---------------------------------------------------------------------------
# module-level factor helpers
# ---------------------------------------------------------------------------
def _factor_laurent(datum, alpha, k, scalar=None) -> TorusLaurent:
    """`1 − 𝖖^k v^α` as a TorusLaurent (α need not be positive)."""
    alpha = tuple(alpha)
    return TorusLaurent(datum, {
        (0,) * datum.dim: _s_const(scalar, 1),
        alpha: _s_qmono(scalar, int(k), -1),
    }, scalar)


def _den_laurent(datum, den: dict, scalar=None) -> TorusLaurent:
    out = TorusLaurent.one(datum, scalar)
    for (a, k), m in den.items():
        out = out * (_factor_laurent(datum, a, k, scalar) ** m)
    return out


# ===========================================================================
# WeylTorusRing — the front door
# ===========================================================================
class WeylTorusRing:
    """The 𝖖-rational torus ring over a `RootDatum`.  `scalar=None` ⇒ Z[𝖖^±];
    `scalar=R` (a ZPlusRing) ⇒ R[𝖖^±] coefficients (flavour in the coefficients)."""

    def __init__(self, datum: RootDatum, scalar=None):
        self.datum = datum
        self.scalar = scalar

    @property
    def dim(self): return self.datum.dim

    def zero(self): return TorusRational.zero(self.datum, self.scalar)

    def one(self): return TorusRational.one(self.datum, self.scalar)

    def from_scalar(self, s): return TorusRational.from_scalar(self.datum, s, self.scalar)

    def monomial(self, weight, lp=None):
        return TorusRational.from_laurent(
            TorusLaurent.monomial(self.datum, weight, lp, self.scalar))

    def laurent(self, terms):
        return TorusLaurent(self.datum, terms, self.scalar)

    def factor(self, alpha, k):
        """`1 − 𝖖^k v^α`."""
        return TorusRational.factor(self.datum, alpha, k, self.scalar)

    def factor_inv(self, alpha, k):
        """`1 / (1 − 𝖖^k v^α)`."""
        return TorusRational.factor_inv(self.datum, alpha, k, self.scalar)

    def __repr__(self):
        return f"WeylTorusRing({self.datum!r}, scalar={self.scalar!r})"
