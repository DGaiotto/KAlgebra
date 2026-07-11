"""Half-integer-exponent Laurent polynomial ring in lq.

Coefficient ring for Le's quantum trace map, R = Z[lq^{1/2}, lq^{-1/2}].

`lq` is Le's q (distinct from the repo's \\fq / Coulomb-algebra q).  The
state-sum projection of skein elements to Y_Delta has coefficients in
this ring; some matrix entries (e.g. the eq.qt formula's lq^{eps eps' / 2})
genuinely require half-integer exponents.

Internal storage: a dict keyed by 2*half-exponent.  So
    lq^{1/2}    is stored under key 1
    lq          is stored under key 2
    lq^{-3/2}   is stored under key -3
This keeps all keys integer and avoids float / Fraction overhead.

Public API exposes half-exponents through accessors that take and
return Fraction (denominator 1 or 2).  Construction helpers accept
either int (interpreted as integer power) or Fraction (interpreted
as half-power if denominator is 2).

Self-contained: no imports from elsewhere in the repo.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Mapping


HalfExp = int | Fraction


def _to_double_key(e: HalfExp) -> int:
    """Convert a half-exponent to its internal 2*e key."""
    if isinstance(e, int):
        return 2 * e
    if isinstance(e, Fraction):
        if e.denominator == 1:
            return 2 * e.numerator
        if e.denominator == 2:
            return e.numerator
        raise ValueError(
            f"exponent must be an integer or half-integer, got {e}"
        )
    raise TypeError(f"exponent must be int or Fraction, got {type(e).__name__}")


def _from_double_key(k: int) -> Fraction:
    """Convert an internal 2*e key back to a Fraction half-exponent."""
    return Fraction(k, 2)


class HalfLaurent:
    """Element of R = Z[lq^{1/2}, lq^{-1/2}].

    Immutable.  Equality, hashing, and arithmetic operate over Z;
    coefficients are stored as Python ints.
    """

    __slots__ = ("_c",)

    def __init__(self, coeffs: Mapping[HalfExp, int] | None = None):
        c: dict[int, int] = {}
        if coeffs is not None:
            for e, v in coeffs.items():
                if v == 0:
                    continue
                k = _to_double_key(e)
                c[k] = c.get(k, 0) + int(v)
                if c[k] == 0:
                    del c[k]
        object.__setattr__(self, "_c", c)

    # construction helpers

    @classmethod
    def zero(cls) -> "HalfLaurent":
        return cls()

    @classmethod
    def one(cls) -> "HalfLaurent":
        return cls({0: 1})

    @classmethod
    def from_int(cls, n: int) -> "HalfLaurent":
        if n == 0:
            return cls.zero()
        return cls({0: int(n)})

    @classmethod
    def monomial(cls, exp: HalfExp, coeff: int = 1) -> "HalfLaurent":
        """coeff * lq^{exp}.  exp may be int or Fraction (denominator 1 or 2)."""
        if coeff == 0:
            return cls.zero()
        return cls({exp: int(coeff)})

    # accessors

    def coefficient(self, exp: HalfExp) -> int:
        return self._c.get(_to_double_key(exp), 0)

    def exponents(self) -> list[Fraction]:
        return [_from_double_key(k) for k in sorted(self._c)]

    def items(self) -> list[tuple[Fraction, int]]:
        return [(_from_double_key(k), v) for k, v in sorted(self._c.items())]

    def is_zero(self) -> bool:
        return not self._c

    def is_integer_polynomial(self) -> bool:
        """True iff every exponent is an integer (no genuine lq^{1/2})."""
        return all(k % 2 == 0 for k in self._c)

    # arithmetic

    def __add__(self, other) -> "HalfLaurent":
        if isinstance(other, int):
            other = HalfLaurent.from_int(other)
        if not isinstance(other, HalfLaurent):
            return NotImplemented
        c = dict(self._c)
        for k, v in other._c.items():
            c[k] = c.get(k, 0) + v
            if c[k] == 0:
                del c[k]
        return _raw(c)

    __radd__ = __add__

    def __neg__(self) -> "HalfLaurent":
        return _raw({k: -v for k, v in self._c.items()})

    def __sub__(self, other) -> "HalfLaurent":
        return self + (-other if isinstance(other, HalfLaurent) else -int(other))

    def __rsub__(self, other) -> "HalfLaurent":
        return (-self) + other

    def __mul__(self, other) -> "HalfLaurent":
        if isinstance(other, int):
            if other == 0:
                return HalfLaurent.zero()
            return _raw({k: v * other for k, v in self._c.items()})
        if not isinstance(other, HalfLaurent):
            return NotImplemented
        out: dict[int, int] = {}
        for k1, v1 in self._c.items():
            for k2, v2 in other._c.items():
                k = k1 + k2
                out[k] = out.get(k, 0) + v1 * v2
                if out[k] == 0:
                    del out[k]
        return _raw(out)

    __rmul__ = __mul__

    def __pow__(self, n: int) -> "HalfLaurent":
        if n < 0:
            # invertible only for monomials; reject the general case
            if len(self._c) != 1:
                raise ValueError("negative power requires a monomial")
            ((k, v),) = self._c.items()
            if abs(v) != 1:
                raise ValueError("negative power requires unit coefficient (+/-1)")
            new_v = 1 if v == 1 else (1 if (-n) % 2 == 0 else -1)
            return _raw({n * k: new_v})
        result = HalfLaurent.one()
        base = self
        e = n
        while e > 0:
            if e & 1:
                result = result * base
            base = base * base
            e >>= 1
        return result

    # shift by a half-exponent (multiply by lq^{shift})

    def shift(self, shift: HalfExp) -> "HalfLaurent":
        if shift == 0 or (isinstance(shift, Fraction) and shift == 0):
            return self
        ds = _to_double_key(shift)
        return _raw({k + ds: v for k, v in self._c.items()})

    # equality / hashing

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            other = HalfLaurent.from_int(other)
        if not isinstance(other, HalfLaurent):
            return NotImplemented
        return self._c == other._c

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._c.items())))

    # display

    def __repr__(self) -> str:
        if not self._c:
            return "0"
        parts: list[str] = []
        for k in sorted(self._c):
            v = self._c[k]
            sign = "-" if v < 0 else "+"
            absv = abs(v)
            if k == 0:
                term = f"{absv}"
            else:
                e = _from_double_key(k)
                if e.denominator == 1:
                    estr = f"{e.numerator}"
                else:
                    estr = f"{e.numerator}/{e.denominator}"
                if absv == 1:
                    term = f"lq^{{{estr}}}"
                else:
                    term = f"{absv}*lq^{{{estr}}}"
            if not parts:
                parts.append(term if sign == "+" else f"-{term}")
            else:
                parts.append(f" {sign} {term}")
        return "".join(parts)

    # specialisation: lq^{1/2} -> 1 (classical limit, returns int)

    def evaluate_at_one(self) -> int:
        return sum(self._c.values())


def _raw(c: dict[int, int]) -> HalfLaurent:
    """Bypass the __init__ normalisation when the dict is already clean."""
    obj = HalfLaurent.__new__(HalfLaurent)
    object.__setattr__(obj, "_c", c)
    return obj


# convenience constants

LQ_HALF = HalfLaurent.monomial(Fraction(1, 2))   # lq^{1/2}
LQ = HalfLaurent.monomial(1)                      # lq
LQ_INV_HALF = HalfLaurent.monomial(Fraction(-1, 2))
LQ_INV = HalfLaurent.monomial(-1)
ZERO = HalfLaurent.zero()
ONE = HalfLaurent.one()
