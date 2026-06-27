"""Exact rational-function-of-q arithmetic for the pure-SU(2) trace.

A `RatFunc` represents an element of `Q(q)` (Laurent in q) as

    q^{shift} * num(q) / den(q),

with `num`, `den` *ordinary* polynomials in q (nonneg powers, nonzero
constant term), gcd-reduced over Q, and `den` normalised to leading
coefficient 1.  This is the carrier for the pure-SU(2) cyclicity trace
solve: every anchor is built as an exact `Q(q)`-combination of the
Wilson seeds and only collapsed to a numeric q-series at the very end,
so there are no truncation buffers anywhere upstream.
"""
from __future__ import annotations

from fractions import Fraction

# A bare polynomial is dict[int>=0, Fraction] with no zero coeffs.


def _p_trim(p: dict) -> dict:
    return {k: v for k, v in p.items() if v != 0}


def _p_mul(a: dict, b: dict) -> dict:
    out: dict = {}
    for i, ca in a.items():
        for j, cb in b.items():
            out[i + j] = out.get(i + j, Fraction(0)) + ca * cb
    return _p_trim(out)


def _p_add(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, Fraction(0)) + v
    return _p_trim(out)


def _p_sub(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, Fraction(0)) - v
    return _p_trim(out)


def _p_scale(a: dict, s: Fraction) -> dict:
    if s == 0:
        return {}
    return {k: v * s for k, v in a.items()}


def _p_divmod(num: dict, den: dict):
    """Polynomial long division over Q.  Returns (quotient, remainder)."""
    num = dict(num)
    den_deg = max(den)
    den_lead = den[den_deg]
    quot: dict = {}
    while num:
        nd = max(num)
        if nd < den_deg:
            break
        c = num[nd] / den_lead
        shift = nd - den_deg
        quot[shift] = quot.get(shift, Fraction(0)) + c
        for k, v in den.items():
            kk = k + shift
            num[kk] = num.get(kk, Fraction(0)) - c * v
            if num[kk] == 0:
                del num[kk]
    return _p_trim(quot), _p_trim(num)


def _p_gcd(a: dict, b: dict) -> dict:
    """Monic GCD of two polynomials over Q (Euclid)."""
    a, b = _p_trim(a), _p_trim(b)
    while b:
        _, r = _p_divmod(a, b)
        a, b = b, r
    if not a:
        return {0: Fraction(1)}
    lead = a[max(a)]
    return {k: v / lead for k, v in a.items()}


class RatFunc:
    """Exact `q^{shift} * num/den` over Q, num/den gcd-reduced, den monic."""

    __slots__ = ("shift", "num", "den")

    def __init__(self, shift: int, num: dict, den: dict):
        num = _p_trim(num)
        den = _p_trim(den)
        if not den:
            raise ZeroDivisionError("RatFunc: zero denominator")
        if not num:
            self.shift, self.num, self.den = 0, {}, {0: Fraction(1)}
            return
        # Pull lowest powers out into the shift so num, den have nonzero
        # constant terms.
        ln = min(num)
        ld = min(den)
        shift = shift + ln - ld
        num = {k - ln: v for k, v in num.items()}
        den = {k - ld: v for k, v in den.items()}
        # gcd-reduce.
        g = _p_gcd(num, den)
        if g != {0: Fraction(1)}:
            num, _ = _p_divmod(num, g)
            den, _ = _p_divmod(den, g)
        # normalise den to monic; push the factor into num.
        dlead = den[max(den)]
        if dlead != 1:
            num = _p_scale(num, Fraction(1) / dlead)
            den = _p_scale(den, Fraction(1) / dlead)
        self.shift, self.num, self.den = shift, num, den

    # -- constructors --
    @staticmethod
    def zero() -> "RatFunc":
        return RatFunc(0, {}, {0: Fraction(1)})

    @staticmethod
    def one() -> "RatFunc":
        return RatFunc(0, {0: Fraction(1)}, {0: Fraction(1)})

    @staticmethod
    def from_laurent(lp) -> "RatFunc":
        """From a `LaurentPoly` (dict-of-coeffs) -> RatFunc (den = 1)."""
        coeffs = lp._coeffs if hasattr(lp, "_coeffs") else lp
        if not coeffs:
            return RatFunc.zero()
        return RatFunc(0, {k: Fraction(v) for k, v in coeffs.items()},
                       {0: Fraction(1)})

    def is_zero(self) -> bool:
        return not self.num

    # -- arithmetic --
    def __add__(self, o: "RatFunc") -> "RatFunc":
        if self.is_zero():
            return o
        if o.is_zero():
            return self
        # bring to common shift s = min(self.shift, o.shift)
        s = min(self.shift, o.shift)
        a_num = {k + (self.shift - s): v for k, v in self.num.items()}
        b_num = {k + (o.shift - s): v for k, v in o.num.items()}
        num = _p_add(_p_mul(a_num, o.den), _p_mul(b_num, self.den))
        den = _p_mul(self.den, o.den)
        return RatFunc(s, num, den)

    def __neg__(self) -> "RatFunc":
        return RatFunc(self.shift, _p_scale(self.num, Fraction(-1)), self.den)

    def __sub__(self, o: "RatFunc") -> "RatFunc":
        return self + (-o)

    def __mul__(self, o: "RatFunc") -> "RatFunc":
        if self.is_zero() or o.is_zero():
            return RatFunc.zero()
        return RatFunc(self.shift + o.shift,
                       _p_mul(self.num, o.num), _p_mul(self.den, o.den))

    def __eq__(self, o) -> bool:
        if not isinstance(o, RatFunc):
            return NotImplemented
        # Functional (exact) equality.  After construction num/den are
        # gcd-reduced with monic den and nonzero constant terms, so two
        # equal functions share (shift, num, den) — but compare via
        # cross-multiplication as a belt-and-braces canonical test.
        if self.shift != o.shift:
            return (self - o).is_zero()
        return self.num == o.num and self.den == o.den

    def __repr__(self) -> str:
        return f"RatFunc(q^{self.shift} * {self.num} / {self.den})"

    # -- evaluation to a q-series --
    def to_series(self, lp_cls, q_max: int):
        """Evaluate as a truncated Laurent series in q (a `lp_cls`), via one
        power-series inversion of `den` (the only series step in the whole
        pipeline).  `lp_cls` is the `LaurentPoly` class."""
        # series inverse of den (den has nonzero constant term => k0=0)
        inv = _series_inverse_poly(self.den, q_max - self.shift + 4)
        prod = _p_mul(self.num, inv)
        out = {k + self.shift: v for k, v in prod.items()
               if k + self.shift <= q_max}
        return lp_cls({k: v for k, v in out.items() if v != 0})


def _series_inverse_poly(den: dict, q_max: int) -> dict:
    """1/den as an ordinary power series (den has nonzero constant term),
    truncated to degree q_max."""
    c0 = den[0]
    # p = (den - c0)/c0
    p = {k: v / c0 for k, v in den.items() if k != 0}
    inv = {0: Fraction(1)}
    term = {0: Fraction(1)}
    while True:
        nxt = {}
        for i, a in term.items():
            for j, b in p.items():
                if i + j <= q_max:
                    nxt[i + j] = nxt.get(i + j, Fraction(0)) - a * b
        nxt = _p_trim(nxt)
        if not nxt:
            break
        for k, v in nxt.items():
            inv[k] = inv.get(k, Fraction(0)) + v
        term = nxt
    return {k: v / c0 for k, v in inv.items()}
